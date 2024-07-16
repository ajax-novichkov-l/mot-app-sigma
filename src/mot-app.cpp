#include <iostream>
#include "dictionary.h"
#include "iniparser.h"   
#include "BYTETracker.h"

#include "image/image_assist.cpp"

using namespace std;

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

    MI_U32 u32ChannelID = 0;
    std::string mFormat = programGonfig.dataType;
    MI_BOOL bRGB = FALSE;
    MI_IPU_SubNet_InputOutputDesc_t desc;
    MI_IPU_TensorVector_t InputTensorVector;
    MI_IPU_TensorVector_t OutputTensorVector;
    MI_IPU_OfflineModelStaticInfo_t OfflineModelInfo;

    BYTETracker tracker(1);//fps = 6
    int num_frames = 0;
    int total_ms = 1;
    
    std::vector<string> class_list;
    ifstream ifs(programGonfig.path_labels.c_str());
    string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
        printf("Class - %s\n", line.c_str());
    }

    static char labels[LABEL_CLASS_COUNT][LABEL_NAME_MAX_SIZE];
    int labelCount = GetLabels(programGonfig.path_labels.c_str() , labels);

MI_SYS_Init();

    //1.create device
    if(MI_SUCCESS != MI_IPU_GetOfflineModeStaticInfo(NULL, (char*)programGonfig.path_model.c_str(), &OfflineModelInfo))
    {
        cout<<"get model variable buffer size failed!"<<std::endl;
        return -1;
    }

    cout<<"Model variable buffer size - " << OfflineModelInfo.u32VariableBufferSize <<std::endl;

    if(MI_SUCCESS !=IPUCreateDevice((char*)programGonfig.path_ipuFw.c_str(),OfflineModelInfo.u32VariableBufferSize))
    {
        cout<<"create ipu device failed!"<<std::endl;
        return -1;
    }



    //2.create channel
    if(MI_SUCCESS!=IPUCreateChannel(&u32ChannelID,(char*)programGonfig.path_model.c_str()))
    {
         cout<<"create ipu channel failed!"<<std::endl;
         MI_IPU_DestroyDevice();
         return -1;
    }


    //3.get input/output tensor

    MI_IPU_GetInOutTensorDesc(u32ChannelID, &desc);
    std::cerr << "Num of outputs == " << desc.u32OutputTensorCount << std::endl;
    std::cerr << "Num of inputs == " << desc.u32InputTensorCount << std::endl;
   /* if (desc.u32OutputTensorCount != 1)
    {
        std::cerr << "Num of output != 1, not 5th yolo!" << std::endl;
        IPUDestroyChannel(u32ChannelID);
        MI_IPU_DestroyDevice();
        return -1;
    }*/

    InputTensorVector.u32TensorCount = desc.u32InputTensorCount;
    if (MI_SUCCESS != IPU_Malloc(&InputTensorVector.astArrayTensors[0], desc.astMI_InputTensorDescs[0].s32AlignedBufSize))
    {
        IPUDestroyChannel(u32ChannelID);
        MI_IPU_DestroyDevice();
        return -1;
    }
    cout<<"s32AlignedBufSize_in :"<< desc.astMI_InputTensorDescs[0].s32AlignedBufSize<<endl;
    OutputTensorVector.u32TensorCount = desc.u32OutputTensorCount;
    for (MI_S32 idx = 0; idx < desc.u32OutputTensorCount; idx++)
    {
        if (MI_SUCCESS != IPU_Malloc(&OutputTensorVector.astArrayTensors[idx], desc.astMI_OutputTensorDescs[idx].s32AlignedBufSize))
        {
            IPUDestroyChannel(u32ChannelID);
            MI_IPU_DestroyDevice();
            return -1;
        }
       std::cout<<"s32AlignedBufSize_out :"<< desc.astMI_OutputTensorDescs[0].s32AlignedBufSize<<std::endl;
       std::cout<<"s32AlignedBufName_out :"<< desc.astMI_OutputTensorDescs[0].name<<std::endl;
       std::cout<<"s32AlignedBufeElmFormat_out :"<< desc.astMI_OutputTensorDescs[0].eElmFormat<<std::endl;
       std::cout<<"s32AlignedBufu32InnerMostStride_out :"<< desc.astMI_OutputTensorDescs[0].u32InnerMostStride<<std::endl;
       std::cout<<"s32AlignedBuffScalar_out :"<< desc.astMI_OutputTensorDescs[0].fScalar<<std::endl;
       std::cout<<"s32AlignedBuffs64ZeroPoint_out :"<< desc.astMI_OutputTensorDescs[0].s64ZeroPoint<<std::endl;
    }

    /*int iResizeH = desc.astMI_InputTensorDescs[0].u32TensorShape[1];
    std::cerr << "iResizeH = "<< iResizeH << std::endl;
    int iResizeW = desc.astMI_InputTensorDescs[0].u32TensorShape[2];
    std::cerr << "iResizeW = "<< iResizeW << std::endl;
    int iResizeC = desc.astMI_InputTensorDescs[0].u32TensorShape[3];
    std::cerr << "iResizeC = "<< iResizeC << std::endl;*/
    //unsigned char *pu8ImageData = new unsigned char[iResizeH*iResizeW*iResizeC];

    PreProcessedData stProcessedData;
    stProcessedData.iResizeH = desc.astMI_InputTensorDescs[0].u32TensorShape[1];
    stProcessedData.iResizeW = desc.astMI_InputTensorDescs[0].u32TensorShape[2];
    stProcessedData.iResizeC = desc.astMI_InputTensorDescs[0].u32TensorShape[3];
    stProcessedData.pdata = (MI_U8 *)InputTensorVector.astArrayTensors[0].ptTensorData[0];
    stProcessedData.pImagePath = programGonfig.path_imgages;
    /*if(strncmp(pRGB,"RGB",sizeof("RGB"))==0)
    {
        bRGB = TRUE;
    }*/

    programGonfig.edgeSizeW = stProcessedData.iResizeW;
    programGonfig.edgeSizeH = stProcessedData.iResizeH;

    int iDimCount = desc.astMI_OutputTensorDescs[0].u32TensorDim;
        //cout<<"iDimCount :" << iDimCount <<std::endl;
    int s32ClassCount  = 1;
    for(int i=0;i<iDimCount;i++ )
    {
        s32ClassCount *= desc.astMI_OutputTensorDescs[0].u32TensorShape[i];
        std::cout<<"the class Count :"<<desc.astMI_OutputTensorDescs[0].u32TensorShape[i]<<std::endl;
        if(i == 1){
            programGonfig.iterations = desc.astMI_OutputTensorDescs[0].u32TensorShape[i];
        }
    }

    if (mFormat.empty()) {
        mFormat = "BGR";
    }
    else if ((mFormat != "BGR") && (mFormat != "RGB") && (mFormat != "BGRA") && (mFormat != "RGBA") &&
            (mFormat != "YUV_NV12") && (mFormat != "GRAY") && (mFormat != "RAWDATA_S16_NHWC") &&
            (mFormat != "DUMP_RAWDATA")) {
        std::cout << "model input format only support `BGR / RGB / BGRA / RGBA / YUV_NV12 / GRAY / RAWDATA_S16_NHWC / DUMP_RAWDATA`" << std::endl;
        return 0;
    }

    stProcessedData.mFormat = mFormat;
    std::cout<<"GetImage format - "<< stProcessedData.mFormat << std::endl;
    //GetImage(&stProcessedData);

    std::string foldername {programGonfig.path_imgages};
    std::vector<std::string> images;
    parse_images_dir(foldername, images);

    std::cout << "Files count - : " << images.size() << std::endl;

for (int idx = 0; idx < images.size(); idx++) {
        std::cout << idx + 1 << " / " << images.size() << '\t';
        stProcessedData.pImagePath = images[idx];

        float reatio = OpenCV_Image(&stProcessedData);
        cout<<"memcpy ok"<<endl;

        //4.invoke
        #if 1
            struct  timeval    tv_start;
            struct  timeval    tv_end;
            gettimeofday(&tv_start,NULL);
        #endif

        //cout<<"IPU invoke :"<<endl;
        int times = 1;
        for (int i=0;i<times;i++ )
        {
            if(MI_SUCCESS!=MI_IPU_Invoke(u32ChannelID, &InputTensorVector, &OutputTensorVector))
            {
                cout<<"IPU invoke failed!!"<<endl;
                //delete pu8ImageData;
                IPUDestroyChannel(u32ChannelID);
                MI_IPU_DestroyDevice();
                return -1;
            }
        }
        #if 1
            gettimeofday(&tv_end,NULL);
            int elasped_time = (tv_end.tv_sec-tv_start.tv_sec)*1000+(tv_end.tv_usec-tv_start.tv_usec)/1000;
            cout<<"fps:"<<1000.0/(float(elasped_time)/times)<<std::endl;
        #endif

        cout<<"show result of detect :"<<endl;

        // show result of detect
        //float *pfData = new float[s32ClassCount];
        float *outDATA = (float*)OutputTensorVector.astArrayTensors[0].ptTensorData[0];
        float fScalar = (float)desc.astMI_OutputTensorDescs[0].fScalar;

        //cout<<"New array size :" << s32ClassCount << std::endl;
        /*for (int i = 0; i < s32ClassCount; i++)
        {
            pfData[i] = *(outDATA + i) * fScalar;
        }*/
        //int16_t *ps16Data = (MI_IPU_FORMAT_INT16*)OutputTensorVector.astArrayTensors[0].ptTensorData[0];//phyTensorAddr//ptTensorData
        cv::Mat frame;
        frame = cv::imread(stProcessedData.pImagePath);

        std::string name = stProcessedData.pImagePath;

        unsigned int pos = stProcessedData.pImagePath.rfind("/");
        if (pos > 0 && pos < stProcessedData.pImagePath.size()) {
            name = name.substr(pos + 1);
        }

        std::string strOutImageName = name;
        strOutImageName = "out_"+strOutImageName;

        //cout<<"checkData \n"<<endl;

        cv::Mat img = checkData(frame, outDATA, class_list, fScalar, strOutImageName, desc.astMI_OutputTensorDescs[0].u32InnerMostStride/getTypeSYze(desc.astMI_OutputTensorDescs[0].eElmFormat), &programGonfig);
        vector<STrack> output_stracks = tracker.update(objects); 
        trackToImage(img, output_stracks, class_list, tracker);

        if(programGonfig.out_boxes){
            cv::imwrite(strOutImageName.c_str(), img);
            cout<<"Save image: "<< strOutImageName.c_str() <<endl;
        }


        if(programGonfig.out_dump){
            std::string dmp_name = strOutImageName;
            unsigned int dmp_pos = dmp_name.rfind(".");
            if (dmp_pos > 0 && dmp_pos < dmp_name.size()) {
                dmp_name = dmp_name.substr(0, dmp_pos);
            }
            dmp_name = dmp_name + ".dmp";

            FILE* fp = fopen(dmp_name.c_str(),"w");
            fwrite((MI_U8*)OutputTensorVector.astArrayTensors[0].ptTensorData[0], 1, desc.astMI_OutputTensorDescs[0].s32AlignedBufSize, fp);
            fclose(fp);
        }
    }

    IPU_Free(&InputTensorVector.astArrayTensors[0], desc.astMI_InputTensorDescs[0].s32AlignedBufSize);
    for (MI_S32 idx = 0; idx < desc.u32OutputTensorCount; idx++)
    {
        IPU_Free(&OutputTensorVector.astArrayTensors[idx], desc.astMI_OutputTensorDescs[idx].s32AlignedBufSize);
    }

    //6.destroy channel/device

    //delete pu8ImageData;
    IPUDestroyChannel(u32ChannelID);
    MI_IPU_DestroyDevice();

    //delete[] pfData;
    iniparser_freedict(pstDict);

    return 0;
}