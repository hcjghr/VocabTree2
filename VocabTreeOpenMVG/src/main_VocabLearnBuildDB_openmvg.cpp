#include <iostream>
#include <cstdlib>
#include <fstream>
#include <algorithm>

#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/sfm/pipelines/sfm_features_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_regions_provider.hpp"

#include "openMVG/stl/stl.hpp"
#include "third_party/cmdLine/cmdLine.h"

#include "VocabTree.h"
#include "VocabTree_OpenMVG.hpp"

using namespace openMVG;
using namespace openMVG::sfm;
using namespace openMVG::features;

int main(int argc, char **argv) 
{
  CmdLine cmd;
 
  std::string sSfM_Data_Filename;
  std::string sMatchesDirectory;
  
  std::string sTreeIn="";  // Vocabulary tree file
  std::string sVocabDB="";  // Vocabulary DB file  
  
  std::string sPairsOutput;
  std::string sScoresOutput;
  
  // Settings Learn
  int depth = 0; //depth of tree. 0 indicates a flat tree
  int bf = 5000; //number of children each non-leaf node.
  int restarts = 1; //number of trials in each run of k-means
  // Settings Build
  bool use_tfidf = true;
  bool normalize = true;
  DistanceType distance_type = DistanceMin;
  
  
  //required
  cmd.add( make_option('i', sSfM_Data_Filename, "input_file") );
  cmd.add( make_option('m', sMatchesDirectory, "feat_dir") );
  
  cmd.add( make_option('t', sTreeIn, "tree_file") );
  
  cmd.add( make_option('d', depth, "tree_depth") );
  cmd.add( make_option('b', bf, "tree_branching") );
  cmd.add( make_option('r', restarts, "kmeans_restart") );
    
  cmd.add( make_option('o', sVocabDB, "out_file") );
  
  try {
    if (argc == 1) throw std::string("Invalid command line parameter.");
    cmd.process(argc, argv);
  } catch(const std::string& s) {
    std::cerr << "Usage: " << argv[0] << '\n'
    << "[-i|--input_file] a SfM_Data file\n"
    << "[-m|--feat_dir] path where features are stored\n"
    
    << "[-t|--tree_file] path where vocabulary tree is stored\n"
    
    << "[-d|--tree_depth] depth of tree. 0 indicates a flat tree [0]\n" 
    << "[-b|--tree_branching] number of children each non-leaf node [5000]\n" 
    << "[-r|--kmeans_restart] number of trials in each run of k-means [1]\n" 
    
    << "[-o|--out_file] file where vocabulary database will be saved\n"       
    << std::endl;

    std::cerr << s << std::endl;
    return EXIT_FAILURE;
  }
  
  std::cout << "\n" << " You called : " << "\n"
    << argv[0] << "\n"
    << "--input_file " << sSfM_Data_Filename << "\n"
    << "--feat_dir " << sMatchesDirectory << "\n"
    << "--tree_file " << sTreeIn << "\n"
    << "--tree_depth " << depth << "\n"
    << "--tree_branching " << bf << "\n"
    << "--kmeans_restart " << restarts << "\n"
    << "--out_file " << sVocabDB << "\n"
    << std::endl;
    
  //---------------------------------------
  // Read SfM Scene (image view data)
  //---------------------------------------
  SfM_Data sfm_data;
  if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(VIEWS))) {
    std::cerr << std::endl
      << "The input SfM_Data file \""<< sSfM_Data_Filename << "\" cannot be read." << std::endl;
    return EXIT_FAILURE;
  }
    
  //---------------------------------------
  // Load SfM Scene regions
  //---------------------------------------
  // Init the regions_type from the image describer file (used for image regions extraction)
  using namespace openMVG::features;
  const std::string sImage_describer = stlplus::create_filespec(sMatchesDirectory, "image_describer", "json");
  std::unique_ptr<Regions> regions_type = Init_region_type_from_file(sImage_describer);
  if (!regions_type)
  {
    std::cerr << "Invalid: "
      << sImage_describer << " regions type file." << std::endl;
    return EXIT_FAILURE;
  }
  
  // Load the corresponding view regions
  std::shared_ptr<Regions_Provider> regions_provider = std::make_shared<Regions_Provider>();
  if (!regions_provider->load(sfm_data, sMatchesDirectory, regions_type)) {
    std::cerr << std::endl << "Invalid regions." << std::endl;
    return EXIT_FAILURE;
  }
  
  
  VocabTree tree;
  
  // Load Vocabulary tree 
  bool treeLoaded;
  treeLoaded = readTree(tree, sTreeIn);
  if(!treeLoaded){
    //-------------------------------------------------
    // LEARN VOCABULARY TREE
    //-------------------------------------------------
    learnVocabularyTree(tree, regions_provider, depth, bf, restarts);
    // Save to file
    if(!sTreeIn.empty()){
      bool bSaveTree = saveTreeToFile(tree,sTreeIn);
      if (bSaveTree){
        std::cout<<"[VocabTree::Build] Save OK"<<std::endl;
      }
      else{ 
        std::cout<<"[VocabTree::Build] Save ERROR"<<std::endl;
      }
    }
  }
  
  //-------------------------------------------------
  // BUILD DB
  //-------------------------------------------------
  buildVocabularyDB(tree,regions_provider,use_tfidf,normalize,distance_type);
  // Save to file
  if(!sVocabDB.empty()){
    bool bSaveTree = saveTreeToFile(tree,sVocabDB);
    if (bSaveTree){
      std::cout<<"[VocabBuildDB] Save OK"<<std::endl;
    }
    else{ 
      std::cout<<"[VocabBuildDB] Save ERROR"<<std::endl;
    }
  }

}



