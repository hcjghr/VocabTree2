#include <iostream>
#include <cstdlib>
#include <fstream>
#include <algorithm>

#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/sfm/pipelines/sfm_regions_provider.hpp"

#include "openMVG/stl/stl.hpp"
#include "third_party/cmdLine/cmdLine.h"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include "VocabTree.h"
#include "qsort.h"

using namespace openMVG;
using namespace openMVG::cameras;
using namespace openMVG::sfm;
using namespace openMVG::features;

#define MAX_ARRAY_SIZE 8388608 // 2 ** 23

void ExportKeypointsLoweSIFTtoFile(const SIFT_Regions* region_i,const std::string fname){
  ofstream sift_file;
  // Open file
  sift_file.open(fname+".txt");

  int num = region_i->GetRegionsPositions().size();
  
  // Add to file the number of features and length of descriptor
  sift_file<<num<<" "<<128<<"\n";

  for (int i = 0; i < num; i++) {
    const SIOPointFeature point = region_i->Features()[i];
    // Export point data
    sift_file << point.x() 
      << " " << point.y() 
      << " " << point.scale()
      << " " << point.orientation() << "\n";
    // Export 128bit descriptor in 7 lines (6*20 + 8)
    for(int k = 0; k < 128; k++){
      if(k!=0 && k%20==0)
        sift_file<<"\n";
      else
        sift_file<<" ";
      
      sift_file << " " << (short int) region_i->Descriptors().at(i)[k];
    }
    sift_file<<"\n";
  }
  sift_file.close();
}


void ExportPairsToFile(Pair_Set &expected_pairs,std::string fname){
  ofstream pairs_file;

  // Open file
  pairs_file.open(fname);

  Pair_Set::iterator it;
  for(it = expected_pairs.begin();it!=expected_pairs.end();it++){
    pairs_file << (*it).first << " " << (*it).second <<"\n";
//    printf("I: %d J: %d\n",(*it).first,(*it).second);
  }
  
  // Close file
  pairs_file.close();
}

void ExportScoreToFile(float* score_matrix, unsigned int num_db_images,std::string fname){
  ofstream pairs_file;

  // Open file
  pairs_file.open(fname);

  for(unsigned int i=0;i<num_db_images;i++){
    for(unsigned int j=0;j<num_db_images;j++){
      pairs_file << score_matrix[i*num_db_images + j];
      if(j==num_db_images-1)
        pairs_file << "\n";
      else
        pairs_file << " ";
    }
  }

  // Close file
  pairs_file.close();
}




int readKeys(const SIFT_Regions* region_i, unsigned char* &keys){
  const int dim = 128;
  int num_keys = region_i->GetRegionsPositions().size();
  if (num_keys > 0) {
    keys = new unsigned char[num_keys * dim];
    std::memcpy((void*)(keys),(void*)region_i->DescriptorRawData(),num_keys*dim*sizeof(unsigned char));
  }
  else{
    keys=NULL;
  }
  return num_keys;
}


void learnVocabularyTree(VocabTree &tree, std::shared_ptr<Regions_Provider> regions_provider, int depth, int bf, int restarts ){
  //-------------------------------------------------
  // LEARN VOCABULARY TREE
  //-------------------------------------------------
  const int dim = 128;
  
  const unsigned int nb_imgs = regions_provider->regions_per_view.size();

  unsigned long total_keys = 0;
  for(unsigned int i=0;i<nb_imgs;i++){
    total_keys+=regions_provider->regions_per_view.at(i)->GetRegionsPositions().size();
  }
  printf("Total number of keys: %lu\n", total_keys);

  // Reduce the branching factor if need be if there are not
  // enough keys, to avoid problems later.
  if (bf >= (int)total_keys){
    bf = total_keys - 1;
    //printf("Reducing the branching factor to: %d\n", bf);
  }
  //fflush(stdout);

  unsigned long long len = (unsigned long long) total_keys * dim;
  int num_arrays =
      len / MAX_ARRAY_SIZE + ((len % MAX_ARRAY_SIZE) == 0 ? 0 : 1);


  unsigned char **vs = new unsigned char *[num_arrays];
//  printf("Allocating %llu bytes in total, in %d arrays\n", len, num_arrays);

  unsigned long long total = 0;
  for (int i = 0; i < num_arrays; i++) {
      unsigned long long remainder = len - total;
      unsigned long len_curr = std::min<unsigned long long>(remainder, MAX_ARRAY_SIZE);
//      printf("Allocating array of size %lu\n", len_curr);
//      fflush(stdout);
      vs[i] = new unsigned char[len_curr];
      remainder -= len_curr;
  }

  /* Create the array of pointers */
//  printf("Allocating pointer array of size %lu\n", 4 * total_keys);
//  fflush(stdout);
  unsigned char **vp = new unsigned char *[total_keys];

  unsigned long off = 0;
  unsigned long curr_key = 0;
  int curr_array = 0;

  C_Progress_display my_progress_bar( nb_imgs,
      std::cout, "\n- Copying keypoints -\n");
      
  
  unsigned char *keys;
  int num_keys = 0;
  
  for (unsigned int i = 0; i < nb_imgs; i++) {
    const SIFT_Regions* region_i = dynamic_cast<SIFT_Regions*>(regions_provider->regions_per_view.at(i).get());
    num_keys = readKeys(region_i,keys);

    //ExportKeypointsLoweSIFT(region_i,vec_fileNames.at(i));

    if (num_keys > 0) {
      for (int j = 0; j < num_keys; j++) {
          std::memcpy((void*)(vs[curr_array]+off),(void*)(keys+(j*dim)),dim*sizeof(unsigned char));

          vp[curr_key] = vs[curr_array] + off;
          curr_key++;
          off += dim;
          if (off == MAX_ARRAY_SIZE) {
              off = 0;
              curr_array++;
          }
      }
      delete [] keys;
    }
    ++my_progress_bar;
  }
  std::cout<<"\n";
  std::cout<<"[VocabTree::Build] Start building tree"<<std::endl;
  tree.Build(total_keys, dim, depth, bf, restarts, vp);
}



void buildVocabularyDB(VocabTree &tree, const std::shared_ptr<Regions_Provider> regions_provider, const bool use_tfidf, const bool normalize, const DistanceType distance_type){
  //-------------------------------------------------
  // BUILD DB
  //-------------------------------------------------
  int start_id = 0;

  switch (distance_type) {
  case DistanceDot:
      printf("[VocabMatch] Using distance Dot\n");
      break;
  case DistanceMin:
      printf("[VocabMatch] Using distance Min\n");
      break;
  default:
      printf("[VocabMatch] Using no known distance!\n");
      break;
  }

//  printf("[VocabBuildDB] Reading tree ...\n");
//  fflush(stdout);

#if 1
  tree.Flatten();
#endif

  tree.m_distance_type = distance_type;
  tree.SetInteriorNodeWeight(0.0);

  /* Initialize leaf weights to 1.0 */
  tree.SetConstantLeafWeights();

  const unsigned int num_db_images = regions_provider->regions_per_view.size();

  tree.ClearDatabase();

  unsigned char *keys;
  int num_keys = 0;
  for (unsigned int i = 0; i < num_db_images; i++) {
    const SIFT_Regions* region_i = dynamic_cast<SIFT_Regions*>(regions_provider->regions_per_view.at(i).get());

    num_keys = readKeys(region_i,keys);

    printf("[VocabBuildDB] Adding vector %d (%d keys)\n", start_id + i, num_keys);
    tree.AddImageToDatabase(start_id + i, num_keys, keys);

    if (num_keys > 0)
      delete [] keys;
  }

  if (use_tfidf)
      tree.ComputeTFIDFWeights(num_db_images);

  if (normalize)
      tree.NormalizeDatabase(start_id, num_db_images);

}


bool estimateMatch(VocabTree &tree, const unsigned int num_db_images, const bool normalize, unsigned char* keys, int num_keys, float *scores){
  // Make sure the score is clear
  for (unsigned int j = 0; j < num_db_images; j++)
    scores[j] = 0.0;
    
  if(num_keys>0){
   tree.ScoreQueryKeys(num_keys, normalize, keys, scores);
   return true;
   }
   return false;
}


// score matrix has to be preallocated with size of num_db_images x num_db_images
void findEstimatedPairs(VocabTree &tree,const bool normalize, const float min_match, const std::shared_ptr<Regions_Provider> regions_provider,Pair_Set &expected_pairs, float* score_matrix){
  
  const unsigned int num_db_images = regions_provider->regions_per_view.size();
  // Prepare scores
  float *scores = new float[num_db_images];
  double *scores_d = new double[num_db_images];
  int *perm = new int[num_db_images];
  unsigned char *keys;   
  // Loop through images and find matches  
  C_Progress_display my_progress_bar( num_db_images,
      std::cout, "\n- Finding expected image pairs -\n");
  for(unsigned int i=0; i<num_db_images;i++){
    const SIFT_Regions* region_i = dynamic_cast<SIFT_Regions*>(regions_provider->regions_per_view.at(i).get());
    int num_keys = readKeys(region_i,keys);
    
    if (num_keys>0){
      if(estimateMatch(tree, num_db_images, normalize, keys, num_keys, scores)){
        /* Sort and find the top scores */
        for (unsigned int j = 0; j < num_db_images; j++) {
          scores_d[j] = (double) scores[j];
        }
        
        qsort_descending();
        qsort_perm(num_db_images, scores_d, perm);

        // Image i has similarity to image perm[j] in score_d[j]    
        for(unsigned int j=0; j<num_db_images;j++){
          if(scores_d[j]<min_match)
            break;
          if((int)i<perm[j]){
            expected_pairs.insert(std::make_pair(i,perm[j]));
//            expected_pairs.emplace_back(std::pair<IndexT,IndexT>(i,perm[j]));
          }
        }
       
        // Save results to score matrix        
        if(score_matrix!=NULL){
          for(unsigned int j=0; j<num_db_images;j++){
            score_matrix[i*num_db_images + j] = scores[j]; 
          }
        }
        
      }
      delete [] keys;       
    }
    ++my_progress_bar;    
  }
  std::cout<<"\n";
  
  delete [] scores;
  delete [] scores_d;
  delete [] perm;   
}

void prepareDBForQuery(VocabTree &tree, DistanceType &query_distance_type){
  tree.Flatten();
  tree.SetDistanceType(query_distance_type);
  tree.SetInteriorNodeWeight(0, 0.0);
}




bool readTree(VocabTree &tree,std::string &file_path){
  if(!file_path.empty() && stlplus::is_file(file_path)){
    try {
      tree.Read(file_path.c_str());
    }
    catch(const std::string& s){
      return false;
    }      
  }
  else{
   return false;
  }
  return true;
}

bool saveTreeToFile(VocabTree &tree,std::string &file_path){
  if(!file_path.empty()){
    try {
      tree.Write(file_path.c_str());
    }
    catch(const std::string& s){
      return false;
    }      
  }
  else{
   return false;
  }
  return true;
}


