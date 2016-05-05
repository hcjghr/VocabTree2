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
  
  std::string sDB_Data_Filename;  
  std::string sDBMatchesDirectory;
  std::string sVocabDB="";  // Vocabulary DB file  
  std::string sPairsOutput;
  std::string sScoresOutput;

  bool normalize = true;
  DistanceType distance_type = DistanceMin;  
  float fMatchThreshold = 0.4;
  
  
  //required
  cmd.add( make_option('i', sDB_Data_Filename, "db_input_file") );
  cmd.add( make_option('m', sDBMatchesDirectory, "db_feat_dir") );
  cmd.add( make_option('v', sVocabDB, "db_file") );
  cmd.add( make_option('f', fMatchThreshold, "min_match_thresh") );
  cmd.add( make_option('o', sPairsOutput, "out_file") );
  cmd.add( make_option('s', sScoresOutput, "score_file") );
  
  try {
    if (argc == 1) throw std::string("Invalid command line parameter.");
    cmd.process(argc, argv);
  } catch(const std::string& s) {
    std::cerr << "Usage: " << argv[0] << '\n'
    << "[-i|--db_input_file] a DB_Data file\n"
    << "[-m|--db_feat_dir path] path where db features are stored\n"
    << "[-v|--db_file] vocabulary db file\n"
    << "[-f|--min_match_thresh] min value needed for the pair to be consider possible\n"    
    << "[-o|--out_file] file where expected pairs are saved\n"    
    << "[-s|--score_file] file where score matrix is saved\n"    
    << std::endl;

    std::cerr << s << std::endl;
    return EXIT_FAILURE;
  }
  
  std::cout << " You called : " << "\n"
    << argv[0] << "\n"
    << "--db_input_file " << sDB_Data_Filename << "\n"
    << "--db_feat_dir " << sDBMatchesDirectory << "\n"
    << "--db_file " << sVocabDB << "\n"
    << "--min_match_thresh " << fMatchThreshold << "\n"
    << "--out_file " << sScoresOutput << "\n"
    << "--score_file " << sScoresOutput << "\n"
    << std::endl;
    
  //---------------------------------------
  // Read SfM Scene (image view data)
  //---------------------------------------
  SfM_Data db_data;
  if (!Load(db_data, sDB_Data_Filename, ESfM_Data(VIEWS))) {
    std::cerr << std::endl
      << "The input DB_Data file \""<< sDB_Data_Filename << "\" cannot be read." << std::endl;
    return EXIT_FAILURE;
  }
    
  //---------------------------------------
  // Load SfM Scene regions
  //---------------------------------------
  // Init the regions_type from the image describer file (used for image regions extraction)
  using namespace openMVG::features;
  const std::string sImage_describer = stlplus::create_filespec(sDBMatchesDirectory, "image_describer", "json");
  std::unique_ptr<Regions> regions_type = Init_region_type_from_file(sImage_describer);
  if (!regions_type)
  {
    std::cerr << "Invalid: "
      << sImage_describer << " regions type file." << std::endl;
    return EXIT_FAILURE;
  }
  
  // Load the corresponding view regions
  std::shared_ptr<Regions_Provider> regions_provider = std::make_shared<Regions_Provider>();
  if (!regions_provider->load(db_data, sDBMatchesDirectory, regions_type)) {
    std::cerr << std::endl << "Invalid regions." << std::endl;
    return EXIT_FAILURE;
  }
  
  
  //-------------------------------------------------
  // 
  //-------------------------------------------------
  VocabTree tree;

  bool treeLoaded = readTree(tree, sVocabDB);
  if(!treeLoaded){
    std::cerr << std::endl
          << "The file \""<< sVocabDB << "\" cannot be read." << std::endl;
    return EXIT_FAILURE;
  }

  // Prepare for Query
  prepareDBForQuery(tree, distance_type);
  
  // Prepare result structure
  const unsigned int num_db_images = regions_provider->regions_per_view.size();
  float* score_matrix = new float[num_db_images*num_db_images];
  Pair_Set expected_pairs;
  
  findEstimatedPairs(tree,normalize,fMatchThreshold, regions_provider, expected_pairs,score_matrix);

  if(!sPairsOutput.empty()){
    std::cout << "Exporting expected image pairs to file" << std::endl << std::endl;
    ExportPairsToFile(expected_pairs,sPairsOutput);
  }
  
  if(score_matrix!=NULL && !sScoresOutput.empty()){
    std::cout << "Exporting score matrix to file" << std::endl << std::endl;
    ExportScoreToFile(score_matrix,num_db_images,sScoresOutput);
  }
  


}



