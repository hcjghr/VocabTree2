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
  std::string sVocabularyDBOut="";  // Path for saving the vocabulary DB


  bool use_tfidf = true;
  bool normalize = true;
  DistanceType distance_type = DistanceMin;  
  
  //required
  cmd.add( make_option('i', sSfM_Data_Filename, "input_file") );
  cmd.add( make_option('m', sMatchesDirectory, "feat_dir") );
  cmd.add( make_option('t', sTreeIn, "tree_file") );
  cmd.add( make_option('o', sVocabularyDBOut, "out_file") );
  
  try {
    if (argc == 1) throw std::string("Invalid command line parameter.");
    cmd.process(argc, argv);
  } catch(const std::string& s) {
    std::cerr << "Usage: " << argv[0] << '\n'
    << "[-i|--input_file] a SfM_Data file\n"
    << "[-m|--feat_dir path] path where features are stored\n"
    << "[-t|--tree_file] path where vocabulary tree is stored\n"    
    << "[-o|--out_file] path where vocabulary DB will be stored\n"    
    << std::endl;

    std::cerr << s << std::endl;
    return EXIT_FAILURE;
  }
  
  std::cout << " You called : " << "\n"
    << argv[0] << "\n"
    << "--input_file " << sSfM_Data_Filename << "\n"
    << "--feat_dir " << sMatchesDirectory << "\n"
    << "--tree_file " << sTreeIn << "\n"
    << "--out_file " << sVocabularyDBOut << "\n"
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
  
  
  //-------------------------------------------------
  // BUILD DB
  //-------------------------------------------------
  VocabTree tree;
  
  bool treeLoaded = readTree(tree, sTreeIn);
  if(!treeLoaded){
    std::cerr << std::endl
          << "The file \""<< sTreeIn << "\" cannot be read." << std::endl;
    return EXIT_FAILURE;
  }
  
  buildVocabularyDB(tree,regions_provider,use_tfidf,normalize,distance_type);
  
  // Save vocabulary tree
  if(!sVocabularyDBOut.empty()){
    bool bSaveTree = saveTreeToFile(tree,sVocabularyDBOut);
    if (bSaveTree){
      std::cout<<"[VocabBuildDB] Save OK"<<std::endl;
    }
    else{ 
      std::cout<<"[VocabBuildDB] Save ERROR"<<std::endl;
    }
  }
     
  
}



