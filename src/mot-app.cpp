#include <iostream>
#include "dictionary.h"
#include "iniparser.h"   

using namespace std;

typedef struct _globalConfig{
    std::string path_ipuFw;
    std::string path_model;
    std::string path_imgages;
    std::string path_labels;
    std::string dataType;
    unsigned int edgeSizeH;
    unsigned int edgeSizeW;
    unsigned int iterations;
    bool out_boxes;
    bool out_dump;
    bool out_txt;
    bool out_nms;
    double  threshold_confidence;
    double  threshold_main;
    double  threshold_boxes;
}globalConfig;

globalConfig programGonfig;

struct PreProcessedData {
    std::string pImagePath;
    unsigned int iResizeH;
    unsigned int iResizeW;
    unsigned int iResizeC;
    std::string mFormat;
    unsigned char* pdata;

} ; 


int main(int argc,char *argv[]){
    cout << "Hello there...";
    dictionary *pstDict;
    if ( argc < 1 )
    {
        std::cout << "USAGE: " << argv[0] <<": <ini_file_path>"<<std::endl;
        exit(0);
    } else {
        pstDict = iniparser_load(argv[1]);
        if(pstDict == NULL){
            std::cout << "There is no iniFile!"<<std::endl;
            return -1;
        }
    }       

    programGonfig.path_ipuFw = iniparser_getstring(pstDict, ":path_ipuFw", "path_ipuFw");
    cout << "path_ipuFw - " << programGonfig.path_ipuFw.c_str() << endl;
    programGonfig.path_model = iniparser_getstring(pstDict, ":path_model", "path_model");
    cout << "path_model - " << programGonfig.path_model.c_str() << endl;
    programGonfig.path_imgages = iniparser_getstring(pstDict, ":path_imgages", "path_imgages");
    cout << "path_imgages - " << programGonfig.path_imgages.c_str() << endl;
    programGonfig.path_labels = iniparser_getstring(pstDict, ":path_labels", "path_labels");
    cout << "path_labels - " << programGonfig.path_labels.c_str() << endl;
    programGonfig.dataType = iniparser_getstring(pstDict, ":dataType", "YUV_NV12");
    cout << "dataType - " << programGonfig.dataType.c_str() << endl;

    programGonfig.out_boxes = iniparser_getboolean(pstDict, ":out_boxes", true);
    cout << "out_boxes - " << (programGonfig.out_boxes ? "true" : "false") << endl;
    programGonfig.out_dump = iniparser_getboolean(pstDict, ":out_dump", false);
    cout << "out_dump - " << (programGonfig.out_dump ? "true" : "false") << endl;
    programGonfig.out_txt = iniparser_getboolean(pstDict, ":out_txt", false);
    cout << "out_txt - " << (programGonfig.out_txt ? "true" : "false") << endl;
    programGonfig.out_nms = iniparser_getboolean(pstDict, ":out_nms", false);
    cout << "out_nms - " << (programGonfig.out_nms ? "true" : "false") << endl;


    programGonfig.threshold_confidence = iniparser_getdouble(pstDict, ":threshold_confidence", 0.4);
    cout << "threshold_confidence - " << programGonfig.threshold_confidence  << endl;
    programGonfig.threshold_main = iniparser_getdouble(pstDict, ":threshold_main", 0.4);
    cout << "threshold_main - " <<  programGonfig.threshold_main  << endl;
    programGonfig.threshold_boxes = iniparser_getdouble(pstDict, ":threshold_boxes", 0.7);
    cout << "threshold_boxes - " << programGonfig.threshold_boxes  << endl; 

    return 0;
}