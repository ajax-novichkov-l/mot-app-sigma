#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <mi_sys.h>
#include <fcntl.h>
#include <signal.h>
#include <stdbool.h>
#include <error.h>
#include <errno.h>
#include <pthread.h>
#include <dirent.h>


#include <limits>

#include <string.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
//#include <vector>

#include <string>

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sys/time.h>
#include <unistd.h>

#include "dictionary.h"
#include "iniparser.h"

using namespace cv::dnn;
using namespace cv;
/*using namespace std;

using namespace std;
;*/

#if 0
using std::cout;
using std::endl;
using std::ostringstream;
using std::vector;
using std::string;
using std::ios;
#endif

#include "mi_common_datatype.h"
#include "mi_sys_datatype.h"
#include "mi_ipu.h"
#include "mi_sys.h"
#include "dataType.h"


#define  DETECT_IMAGE_FUNC_INFO(fmt, args...)           do {printf("[Info ] [%-4d] [%10s] ", __LINE__, __func__); printf(fmt, ##args);} while(0)

#ifdef ALIGN_UP
#undef ALIGN_UP
#define ALIGN_UP(x, align) (((x) + ((align) - 1)) & ~((align) - 1))
#else
#define ALIGN_UP(x, align) (((x) + ((align) - 1)) & ~((align) - 1))
#endif
#ifndef ALIGN_DOWN
#define ALIGN_DOWN(val, alignment) (((val)/(alignment))*(alignment))
#endif

#define LABEL_CLASS_COUNT (100)
#define LABEL_NAME_MAX_SIZE (60)

#define FONT_SCALE  0.7f
#define FONT_FACE  cv::FONT_HERSHEY_SIMPLEX
#define  THICKNESS  1
#define SCORE_THRESHOLD 0.4f
#define NMS_THRESHOLD   0.45f
#define DIVP_ALIGNMENT (4)
#define DIVP_YUV_ALIGNMENT (16)
#define SRC_IMAGE_SIZE (1920*1080*4)
#define DST_IMAGE_SIZE (1920*1080*4)   

globalConfig programGonfig;

struct PreProcessedData {
    std::string pImagePath;
    unsigned int iResizeH;
    unsigned int iResizeW;
    unsigned int iResizeC;
    std::string mFormat;
    unsigned char* pdata;

} ; 

struct DetectionBBoxInfo {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int   classID;
}; 

MI_IPU_CHN FdaChn, FrChn;
MI_IPU_SubNet_InputOutputDesc_t FdaDesc, FrDesc;
static MI_IPU_DevAttr_t ipudev;
MI_IPUChnAttr_t FdaChnAttr, FrChnAttr;

typedef struct {
	MI_IPU_DevAttr_t *dev;
	MI_IPUChnAttr_t	 ipuchn;
	MI_IPU_CHN       chn;
	MI_IPU_SubNet_InputOutputDesc_t des;
    MI_IPU_TensorVector_t ipuov;
    int				 indepth;
    int				 outdepth;
	unsigned int	 bufsize;
} ipu_set;

#define	NUMBER_CHANNEL	(4)
static int 				_fwrdy  = 0;
static unsigned int		_chnmap = 0;
static MI_IPU_DevAttr_t _ipudev;


// Colors.
cv::Scalar BLACK = cv::Scalar(0,0,0);
cv::Scalar BLUE = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED = cv::Scalar(0,0,255);

std::vector<Object> objects;

void DIVPCreate(MI_PHY& phySrcBufAddr, MI_PHY& phyDstBufAddr, void* &pVirSrcBufAddr, void* &pVirDstBufAddr) {
    MI_S32 ret;
    ret = MI_SYS_MMA_Alloc(NULL, SRC_IMAGE_SIZE, &phySrcBufAddr);
    if(ret != MI_SUCCESS)
    {
        std::cerr << "alloc src buff failed" << std::endl;
    }

    ret = MI_SYS_Mmap(phySrcBufAddr, SRC_IMAGE_SIZE, &pVirSrcBufAddr, TRUE);
    if(ret != MI_SUCCESS)
    {
        MI_SYS_MMA_Free(phySrcBufAddr);
        std::cerr << "mmap src buff failed" << std::endl;
    }

    ret = MI_SYS_MMA_Alloc(NULL, DST_IMAGE_SIZE, &phyDstBufAddr);
    if(ret != MI_SUCCESS)
    {
        MI_SYS_Munmap(pVirSrcBufAddr, SRC_IMAGE_SIZE);
        MI_SYS_MMA_Free(phySrcBufAddr);
        std::cerr << "alloc dst buff failed" << std::endl;
    }

    ret = MI_SYS_Mmap(phyDstBufAddr, DST_IMAGE_SIZE, &pVirDstBufAddr, TRUE);
    if(ret != MI_SUCCESS)
    {
        MI_SYS_Munmap(pVirSrcBufAddr, SRC_IMAGE_SIZE);
        MI_SYS_MMA_Free(phySrcBufAddr);
        MI_SYS_MMA_Free(phyDstBufAddr);
        std::cerr << "mmap dst buff failed" << std::endl;
    }
}

void DIVPDestory(MI_PHY& phySrcBufAddr, MI_PHY& phyDstBufAddr, void* &pVirSrcBufAddr, void* &pVirDstBufAddr) {
    MI_SYS_Munmap(pVirSrcBufAddr, SRC_IMAGE_SIZE);
    MI_SYS_Munmap(pVirDstBufAddr, DST_IMAGE_SIZE);
    MI_SYS_MMA_Free(phySrcBufAddr);
    MI_SYS_MMA_Free(phyDstBufAddr);
}

void* ipu_init2(int in, int out, unsigned int bsize)
{
	int			err;
	ipu_set		*pu;
	int			ch;

	pu = (ipu_set*)malloc(sizeof(ipu_set));
	memset(pu, 0, sizeof(*pu));
	pu->indepth    = in;
	pu->outdepth   = out;
	if (!_fwrdy) {
		const char	*firmware;
		pu->bufsize = 5 * 1024 * 1024;
    	_ipudev.u32MaxVariableBufSize		= pu->bufsize; // REVIEW?
    	_ipudev.u32YUV420_W_Pitch_Alignment	= 16;
    	_ipudev.u32YUV420_H_Pitch_Alignment	= 2;
    	_ipudev.u32XRGB_W_Pitch_Alignment	= 16;
		if (!(firmware = getenv("IPU_FIRMWARE")))
			firmware = NULL;	//IPU_FIRMWARE;
		if ((firmware && access(firmware, O_RDONLY) == -1) ||
			MI_IPU_CreateDevice(&_ipudev, NULL, (char*)firmware, 0)) {
			free(pu);
			return NULL;
		}
		_fwrdy = 1;
	}
	pu->dev = &_ipudev;
	pu->chn = -1;	// assigned chn by ipu_load_smodel(..)
	return (void*)pu;
	/*
	for (ch = 0; ch < NUMBER_CHANNEL; ch++) {
		if ((_chnmap & (1 << ch)) == 0) {
			_chnmap |= (1 << ch);
			pu->dev = &_ipudev;
			pu->chn = ch;
			return (void*)pu;
		}
	}
	*/
}

void ipu_destroy(void * handle)
{
	ipu_set	*pu;
	pu = (ipu_set*)handle;

	MI_IPU_DestroyCHN(pu->chn);
	_chnmap &= ~(1 << pu->chn);
	free(pu);

	// MI_IPU_DestroyDevice();
}


static void get_hwPitch_Paras(MI_U32 *pu32YUV420_H_Pitch, MI_U32 *pu32YUV420_V_Pitch, MI_U32 *pu32XRGB_H_Pitch){
    *pu32YUV420_H_Pitch = 16;
    *pu32YUV420_V_Pitch = 2;
    *pu32XRGB_H_Pitch = 16;
}

cv::Scalar getColor(float prc){
    cv::Scalar outColor;
    float r = 255.0 - 2.55*prc*100.0;
    float g = 2.55*prc*100.0;
    float b = .0;
    outColor = cv::Scalar((MI_U8)b, (MI_U8)g, (MI_U8)r);
    return outColor;
}

//Y = (R * 218  + G * 732  + B * 74) / 1024
//U = (R * -117 + G * -395 + B * 512)/ 1024
//V = (R * 512  + G * -465 + B * -47)/ 1024
static float rgb2yuv_covert_matrix[9] = {218.0/1024, 732.0/1024, 74.0/1024, -117.0/1024, -395.0/1024, 512.0/1024, 512.0/1024, -465.0/1024, -47.0/1024};

float OpenCV_Image(PreProcessedData* pstPreProcessedData){
    float ratio = 1.0;
    std::string filename = pstPreProcessedData->pImagePath;
    cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error! Image doesn't exist!" << std::endl;
        exit(1);
    }

    cv::Mat sample_resized;
    cv::Size inputSize = cv::Size(pstPreProcessedData->iResizeW, pstPreProcessedData->iResizeH);

    std::string name = filename;
    unsigned int pos = filename.rfind("/");
    if (pos > 0 && pos < filename.size()) {
      name = name.substr(pos + 1);
    }

    std::string strOutImageName = name;


    if (img.size() != inputSize) {
        //std::cout << "input size should be :" << pstPreProcessedData->iResizeC << " " << pstPreProcessedData->iResizeH << " " << pstPreProcessedData->iResizeW << std::endl;
		//std::cout << "now input size is :" << img.channels() << " " << img.rows<<" " << img.cols << std::endl;
		//std::cout << "img is going to resize!" << std::endl;

        ratio = std::min((float)inputSize.width / (float)img.size().width, (float)inputSize.height / (float)img.size().height);
        cv::Size outputSize = cv::Size(int(round(img.size().width * ratio)), int(round(img.size().height * ratio)));
        //std::cout << "outputSize w " << outputSize.width <<std::endl;
        //std::cout << "outputSize h " << outputSize.height <<std::endl;
        cv::resize(img, sample_resized, outputSize, 0, 0, cv::INTER_LINEAR);
        cv::Mat image(pstPreProcessedData->iResizeH, pstPreProcessedData->iResizeW, CV_8UC3, cv::Scalar(0, 0, 0));

        sample_resized.copyTo(image(cv::Rect(0,0,sample_resized.cols, sample_resized.rows)));

        sample_resized = image;

    }
    else {
        std::cout << "No resize needed!\n" <<std::endl;
        sample_resized = img;
    }

    cv::Mat sample;
    int imageSize;
    int imageStride;
    if (pstPreProcessedData->mFormat == "RGB") {
        cv::cvtColor(sample_resized, sample, cv::COLOR_BGR2RGB);
        imageSize = pstPreProcessedData->iResizeH * pstPreProcessedData->iResizeW * 3;
        MI_U8* pSrc = sample.data;
        for(int i = 0; i < imageSize; i++)
        {
            *(pstPreProcessedData->pdata + i) = *(pSrc + i);
        }
    }
    else if (pstPreProcessedData->mFormat == "BGRA") {
        cv::cvtColor(sample_resized, sample, cv::COLOR_BGR2BGRA);
        MI_U8* pSrc = sample.data;
        MI_U8* pVirsrc = pstPreProcessedData->pdata;
        imageStride = ALIGN_UP(pstPreProcessedData->iResizeW * 4, DIVP_YUV_ALIGNMENT);
        imageSize = pstPreProcessedData->iResizeH * imageStride;
        for (int i = 0; i < pstPreProcessedData->iResizeH; i++) {
            memcpy(pVirsrc, pSrc, pstPreProcessedData->iResizeW * 4);
            pVirsrc += imageStride;
            pSrc += pstPreProcessedData->iResizeW * 4;
        }
    }
    else if (pstPreProcessedData->mFormat == "RGBA") {
        cv::cvtColor(sample_resized, sample, cv::COLOR_BGR2RGBA);
        MI_U8* pSrc = sample.data;
        MI_U8* pVirsrc = pstPreProcessedData->pdata;
        imageStride = ALIGN_UP(pstPreProcessedData->iResizeW * 4, DIVP_YUV_ALIGNMENT);
        imageSize = pstPreProcessedData->iResizeH * imageStride;
        for (int i = 0; i < pstPreProcessedData->iResizeH; i++) {
            memcpy(pVirsrc, pSrc, pstPreProcessedData->iResizeW * 4);
            pVirsrc += imageStride;
            pSrc += pstPreProcessedData->iResizeW * 4;
        }
    }
    else if ((pstPreProcessedData->mFormat == "YUV_NV12") || (pstPreProcessedData->mFormat == "GRAY")) {
        int s32c = 3;
        MI_U32 u32YUV420_H_Pitch = 0, u32YUV420_V_Pitch = 0, u32XRGB_H_Pitch = 0;
        int w = sample_resized.cols;
        int h = sample_resized.rows;

        float * pfSrc = (float *)malloc(w * h * s32c * sizeof(*pfSrc));

        for(int j = 0; j < w * h * 3; j++)
        {
            *(pfSrc + j) = *(sample_resized.data + j);
        }

        get_hwPitch_Paras(&u32YUV420_H_Pitch, &u32YUV420_V_Pitch, &u32XRGB_H_Pitch);
        int w_with_pitch = ALIGN_UP(w, u32YUV420_H_Pitch);
        int h_with_pitch = ALIGN_UP(h, u32YUV420_V_Pitch);

        float *yuv444_packet_buf = (float *)malloc(w_with_pitch * h_with_pitch * s32c * sizeof(*yuv444_packet_buf));
        if (yuv444_packet_buf == NULL)
        {
            std::cerr << "alloc yuv444_packet_buf fail " << std::endl;
            exit(-1);
        }
        for (int i = 0; i < w_with_pitch * h_with_pitch * s32c; ++i)
        {
            yuv444_packet_buf[i] = 0.0;
        }

        float *yuv444_semiplanar_buf = (float *)malloc(w_with_pitch * h_with_pitch * s32c * sizeof(*yuv444_semiplanar_buf));
        if (yuv444_semiplanar_buf == NULL)
        {
            std::cerr << "alloc yuv444_semiplanar_buf fail " << std::endl;
            free(yuv444_packet_buf);
            exit(-1);
        }

        imageSize = w * h * 3 / 2;
        float *trans_yuv_buf = (float *)malloc(imageSize * sizeof(*trans_yuv_buf));
        if (trans_yuv_buf == NULL)
        {
            std::cerr << "alloc trans_yuv_buf fail " << std::endl;
            free(pfSrc);
            free(yuv444_packet_buf);
            free(yuv444_semiplanar_buf);
            exit(-1);
        }

        float *yuv420_buf = (float *)trans_yuv_buf;

        std::cout << "img convert from rgb to YUV444 packed!" << std::endl;

        //convert from rgb to YUV444 packed
        for (int index_h = 0; index_h < h; ++index_h)
        {
            for (int index_w = 0; index_w < w; ++index_w)
            {
                float *in_rgb = (float *)pfSrc + (index_h*w + index_w)*3;
                float *out_yuv444 = yuv444_packet_buf + (index_h*w_with_pitch + index_w)*3;

                float in_rgb0 = 0, in_rgb1 = 0, in_rgb2 = 0;
                in_rgb0 = in_rgb[2];
                in_rgb1 = in_rgb[1];
                in_rgb2 = in_rgb[0];

                out_yuv444[0] = rgb2yuv_covert_matrix[0] * in_rgb0 + rgb2yuv_covert_matrix[1] * in_rgb1 + rgb2yuv_covert_matrix[2] * in_rgb2; //Y
                out_yuv444[1] = rgb2yuv_covert_matrix[3] * in_rgb0 + rgb2yuv_covert_matrix[4] * in_rgb1 + rgb2yuv_covert_matrix[5] * in_rgb2 + 128; //U
                out_yuv444[2] = rgb2yuv_covert_matrix[6] * in_rgb0 + rgb2yuv_covert_matrix[7] * in_rgb1 + rgb2yuv_covert_matrix[8] * in_rgb2 + 128; //V

                //for the horizontal pitch, copy w-1 into w position
                if (((w&1) == 1) && (w_with_pitch > w) && (index_w == (w-1)))
                {
                    memcpy(&out_yuv444[3],  &out_yuv444[0], 3*sizeof(*yuv444_packet_buf));
                }
            }

            //for the vertical pitch, copy row h-1 into row h
            if (h_with_pitch == (h+1) && index_h == (h-1))
            {
                float *out_yuv444_last_row = yuv444_packet_buf + index_h*w_with_pitch*3;
                float *out_yuv444_last_row_pitch = yuv444_packet_buf + (index_h+1)*w_with_pitch*3;

                memcpy(out_yuv444_last_row_pitch, out_yuv444_last_row, 3*w_with_pitch*sizeof(*yuv444_packet_buf));
            }
        }

        std::cout << "convert from YUV444_packet to YUV444 semiplanar" << std::endl;

        //convert from YUV444_packet to YUV444 semiplanar
        for (int index_group = 0; index_group < h_with_pitch*w_with_pitch; ++index_group)
        {
            float *in_yuv444_packed = yuv444_packet_buf + 3*index_group;
            float *out_y_yuv444_planar = yuv444_semiplanar_buf + index_group;
            float *out_uv_yuv444_planar = yuv444_semiplanar_buf + h_with_pitch*w_with_pitch + index_group*2;

            out_y_yuv444_planar[0] =  in_yuv444_packed[0]; //Y
            out_uv_yuv444_planar[0] = in_yuv444_packed[1]; //U
            out_uv_yuv444_planar[1] = in_yuv444_packed[2]; //V
        }

        //convert from YUV444_planar to YUV420 planar
        //all Y
        memcpy(yuv420_buf, yuv444_semiplanar_buf, w_with_pitch*h_with_pitch*sizeof(*yuv420_buf));
        //for UV
        float *uv_yuv420_offset = yuv420_buf + w_with_pitch*h_with_pitch;
        float *uv_yuv444_offset = yuv444_semiplanar_buf + w_with_pitch*h_with_pitch;
        for (int index_h = 0; index_h < h_with_pitch/2; ++index_h)
        {
            for (int index_w = 0; index_w < w_with_pitch; index_w+=2)
            {
                float *in_row1_uv_yuv444_planar = uv_yuv444_offset + (index_h*2) * (w_with_pitch*2) + index_w * 2;
                float *in_row2_uv_yuv444_planar = uv_yuv444_offset + (index_h*2+1) * (w_with_pitch*2) + index_w * 2;
                float *out_uv_yuv420_planar = uv_yuv420_offset + index_h*w_with_pitch + index_w;

                float u = in_row1_uv_yuv444_planar[0] + in_row2_uv_yuv444_planar[0] + in_row1_uv_yuv444_planar[2] + in_row2_uv_yuv444_planar[2];
                float v = in_row1_uv_yuv444_planar[1] + in_row2_uv_yuv444_planar[1] + in_row1_uv_yuv444_planar[3] + in_row2_uv_yuv444_planar[3];

                out_uv_yuv420_planar[0] = u/4.0; //u
                out_uv_yuv420_planar[1] = v/4.0; //v
            }
        }
        std::cout << "copy data - "<< imageSize << std::endl;
        for (int k = 0; k < imageSize; k++)
        {
            *((MI_U8 *)pstPreProcessedData->pdata + k) = (MI_U8)round(*(trans_yuv_buf + k));
        }

        free(yuv444_packet_buf);
        free(yuv444_semiplanar_buf);
        free(pfSrc);
        free(trans_yuv_buf);
    }
    else {
        sample = sample_resized;
        imageSize = pstPreProcessedData->iResizeH * pstPreProcessedData->iResizeW * 3;
        MI_U8* pSrc = sample.data;
        for(int i = 0; i < imageSize; i++)
        {
            *(pstPreProcessedData->pdata + i) = *(pSrc + i);
        }
    }

    MI_SYS_FlushInvCache(pstPreProcessedData->pdata, imageSize);
    return ratio;
}

cv::Size draw_label(cv::Mat& input_image, std::string label, int left, int top, int maxSize, float prc){
    // Display the label at the top of the bounding box.
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, FONT_FACE, 1, THICKNESS, &baseLine);
    cv::Size _label_size = cv::getTextSize(label, FONT_FACE, 1, THICKNESS, &baseLine);
    //printf("label_size.w -%d, label_size.h - %d \n", label_size.width, label_size.height);

    top = max(top, label_size.height);
    // Top left corner.
    double scale = 1.0;
    if(label_size.width>maxSize)
        scale = (double)((double)maxSize/(double)label_size.width);

    if(scale<0.3)
        scale = 0.3;

    //std::cout << "label scale - " << scale << std::endl;

    if((left+label_size.width*scale)>input_image.size().width){
        left = input_image.size().width - label_size.width*scale;
    }
    if(left <=0)
    left=1;
        if(top <=0)
    top=1;

    cv::Point tlc = cv::Point(left, top);

    //printf("tlc.x -%d, tlc.y - %d \n", tlc.x, tlc.y);

    cv::Point brc = cv::Point(left + (int)(label_size.width*scale), top + (int)(label_size.height*scale) + baseLine*scale);

    //printf("brc.x -%d, brc.y - %d \n", brc.x, brc.y);

    //printf("rectangle start\n");
    //cv::rectangle(input_image, tlc, brc, cv::Scalar(255, 255, 255), cv::FILLED);

    
    cv::Mat lab = input_image(cv::Rect(tlc, brc));
    cv::Mat color(lab.size(), CV_8UC3, cv::Scalar(0, 0, 0)); 
    double alpha = 0.5;
    cv::addWeighted(color, alpha, lab, 1.0 - alpha , 0.0, lab);

    //printf("rectangle end\n");
    ///("putText\n");
    cv::putText(input_image, label, cv::Point(left, top + (int)(label_size.height*scale)), FONT_FACE, scale, getColor(prc), THICKNESS);
    label_size.width *= (float)scale;
    label_size.height *= (float)scale;
    return label_size;
}

cv::Mat checkData(cv::Mat &inputImg, float *predictions, const std::vector<std::string> &labels, float ratio, std::string filename, int step, globalConfig *conf ) {
    
    bool isFile = false;
    std::ofstream file;

    float addRatio = 1.0;

    if(inputImg.rows != 0){
        addRatio = std::min((float)conf->edgeSizeW / (float)inputImg.size().width, (float)conf->edgeSizeH / (float)inputImg.size().height);
    }
    //float addRatioW = min((float)conf->edgeSizeW / (float)inputImg.size().width, (float)conf->edgeSizeH / (float)inputImg.size().height);

    //printf("addRatio - %f\n", addRatio);
    float xRatio = 1.0;
    float yRatio = 1.0;
    if(inputImg.rows != 0){
        xRatio = (float)inputImg.cols / ((float)inputImg.cols*addRatio);
        yRatio = (float)inputImg.rows / ((float)inputImg.rows*addRatio);
    }
    float *data = predictions;

    const int rows = conf->iterations;

    std::vector<int> _classId;
    std::vector<float> _confidence;
    std::vector<cv::Rect> _bbox;


    if(conf->out_txt){
        std::string name = filename;
        unsigned int pos = name.rfind(".");
        if (pos > 0 && pos < name.size()) {
            name = name.substr(0, pos);
        }
        name = name + ".txt";
        file.open(name, std::ios_base::out);
        isFile = true;
    }

    //printf("labels size - %d \n", labels.size());

    //objects.resize(rows); //indices.size()


    for (int i = 0; i < rows; i++) {

        //int index = i * dimensions;

        float confidence = *(data + 4) * ratio;

        /*if(confidence > 1.0){
           std::cout << "index - " << i << std::endl;
           std::cout << "confidence - " << confidence << std::endl;
        }*/

        if (confidence > conf->threshold_confidence) {// && (confidence <= 1.0)

            float* new_scores = new float[labels.size()];
            for(int i = 0; i<labels.size(); i++){
                new_scores[i] = *(data + 5 + i)*ratio;
            }
            //float *classes_scores = *(data + 5)*ratio;
            cv::Mat scores(1, labels.size(), CV_32FC1, new_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > conf->threshold_main) {

                _confidence.push_back(confidence);
                _classId.push_back(class_id.x);

                float x = *(data + 0)*ratio;
                float y = *(data + 1)*ratio;
                float w = *(data + 2)*ratio;
                float h = *(data + 3)*ratio;

                int left = int((x - 0.5 * w) * xRatio);//
                int top = int((y - 0.5 * h) * yRatio);//
                int width = int((w) * xRatio);//
                int height = int(h * yRatio);//

                _bbox.push_back(cv::Rect(left, top, width, height));
                if(isFile){
                    file << labels[class_id.x].c_str() << ", confidence - " << confidence << ", left - " << left << ", top - " << top << ", width - " << width << ", height - " << height << std::endl;
                }

                /*objects[i].rect.x = left;
                objects[i].rect.y = top;
                objects[i].rect.width = width;
                objects[i].rect.height = height;  
                objects[i].prob = confidence;
                objects[i].label = class_id.x;*/
            }
        }
        data += step;
    }

    //printf("indices part \n");

    std::vector<int> indices;
    NMSBoxes(_bbox, _confidence, conf->threshold_main, conf->threshold_boxes, indices);
    std::ofstream file_nms;
    if(conf->out_nms){
        std::string name_nms = filename;
        unsigned int pos = name_nms.rfind(".");
        if (pos > 0 && pos < name_nms.size()) {
            name_nms = name_nms.substr(0, pos);
        }
        name_nms = name_nms + ".nms";
        file_nms.open(name_nms, std::ios_base::out);
    }

    //printf("indices size part - %d\n", indices.size());

    objects.resize(indices.size()); 

    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];

        //printf("idx = %d\n", idx);

        cv::Rect box = _bbox[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;

        objects[i].rect.x = left;
        objects[i].rect.y = top;
        objects[i].rect.width = width;
        objects[i].rect.height = height;  
        objects[i].prob = _confidence[idx];
        objects[i].label = _classId[idx];


        //rectangle(inputImg, cv::Point(left, top), cv::Point(left + width, top + height), getColor(_confidence[idx]), THICKNESS);
        if(inputImg.rows != 0){
            std::string label = cv::format("%.2f", _confidence[idx]);
            label = labels[_classId[idx]] + ":" + label;
            draw_label(inputImg, label, left, top-50, width, _confidence[idx]);
        }
        
        if(conf->out_nms){
            file_nms << labels[_classId[idx]] << " " << _confidence[idx] << " " << left << " " << top << " " << width << " " << height << std::endl;
        }
        //printf("Lable top - %d\n", top);
        //printf("Lable top ratio - %f\n", _confidence[idx]);
        //printf("Lable mod top - %d\n", (int)((float)top*(1.0 -_confidence[idx])));
    }
    //printf("indices size part end\n");
    if(conf->out_nms){
        file_nms.close();
    }
    if(isFile){
        file.close();
    }
    //printf("return part \n");
    return inputImg;
}

MI_S32  IPUCreateDevice(char *pFirmwarePath,MI_U32 u32VarBufSize){
    MI_S32 s32Ret = MI_SUCCESS;
    MI_IPU_DevAttr_t stDevAttr;
    stDevAttr.u32MaxVariableBufSize = u32VarBufSize;
    stDevAttr.u32YUV420_W_Pitch_Alignment = 16;
    stDevAttr.u32YUV420_H_Pitch_Alignment = 2;
    stDevAttr.u32XRGB_W_Pitch_Alignment = 16;
    s32Ret = MI_IPU_CreateDevice(&stDevAttr, NULL, pFirmwarePath, 0);
    return s32Ret;
}

MI_S32 IPUCreateChannel(MI_U32 *s32Channel, char *pModelImage){
    MI_S32 s32Ret ;
    MI_SYS_GlobalPrivPoolConfig_t stGlobalPrivPoolConf;
    MI_IPUChnAttr_t stChnAttr;

    //create channel
    memset(&stChnAttr, 0, sizeof(stChnAttr));
    stChnAttr.u32InputBufDepth = 0;
    stChnAttr.u32OutputBufDepth = 0;
    return MI_IPU_CreateCHN(s32Channel, &stChnAttr, NULL, pModelImage);
}

MI_S32 IPUDestroyChannel(MI_U32 s32Channel){
    return MI_IPU_DestroyCHN(s32Channel);
}

MI_S32 IPU_Malloc(MI_IPU_Tensor_t* pTensor, MI_U32 BufSize){
    MI_S32 s32ret = 0;
    MI_PHY phyAddr = 0;
    void* pVirAddr = NULL;
    s32ret = MI_SYS_MMA_Alloc(NULL, BufSize, &phyAddr);
    if (s32ret != MI_SUCCESS)
    {
        std::cerr << "Alloc buffer failed!" << std::endl;
        return s32ret;
    }
    s32ret = MI_SYS_Mmap(phyAddr, BufSize, &pVirAddr, TRUE);
    if (s32ret != MI_SUCCESS)
    {
        std::cerr << "Mmap buffer failed!" << std::endl;
        MI_SYS_MMA_Free(phyAddr);
        return s32ret;
    }
    pTensor->phyTensorAddr[0] = phyAddr;
    pTensor->ptTensorData[0] = pVirAddr;
    return s32ret;
}

MI_S32 IPU_Free(MI_IPU_Tensor_t* pTensor, MI_U32 BufSize){
    MI_S32 s32ret = 0;
    s32ret = MI_SYS_Munmap(pTensor->ptTensorData[0], BufSize);
    s32ret = MI_SYS_MMA_Free(pTensor->phyTensorAddr[0]);
    return s32ret;
}

void GetImage(PreProcessedData *pstPreProcessedData){
    std::string filename=(std::string)(pstPreProcessedData->pImagePath);
    cv::Mat sample;
    cv::Mat img = cv::imread(filename, -1);
    if (img.empty()) {
      std::cout << " error!  image don't exist!" << std::endl;
      exit(1);
    }


    int num_channels_  = pstPreProcessedData->iResizeC;
    if (img.channels() == 3 && num_channels_ == 1)
    {
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    }
    else if (img.channels() == 4 && num_channels_ == 1)
    {
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    }
    else if (img.channels() == 4 && num_channels_ == 3)
    {
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    }
    else if (img.channels() == 1 && num_channels_ == 3)
    {
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    }
    else
    {
        sample = img;
    }

    cv::Mat sample_float;
    if (num_channels_ == 3)
      sample.convertTo(sample_float, CV_32FC3);
    else
      sample.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_norm =sample_float ;
    /*if (pstPreProcessedData->bRGB)
    {
        cv::cvtColor(sample_float, sample_norm, cv::COLOR_BGR2RGB);
    }*/

    cv::Mat sample_resized;
    cv::Size inputSize = cv::Size(pstPreProcessedData->iResizeW, pstPreProcessedData->iResizeH);
    if (sample.size() != inputSize)
    {
		//std::cout << "input size should be :" << pstPreProcessedData->iResizeC << " " << pstPreProcessedData->iResizeH << " " << pstPreProcessedData->iResizeW << std::endl;
		//std::cout << "now input size is :" << img.channels() << " " << img.rows<<" " << img.cols << std::endl;
		//std::cout << "img is going to resize!" << std::endl;
		resize(sample_norm, sample_resized, inputSize);
	}
    else
	{
      sample_resized = sample_norm;
    }

    float *pfSrc = (float *)sample_resized.data;
    int imageSize = pstPreProcessedData->iResizeC*pstPreProcessedData->iResizeW*pstPreProcessedData->iResizeH;

    for(int i=0;i<imageSize;i++)
    {
        *(pstPreProcessedData->pdata+i) = (unsigned char)(round(*(pfSrc + i)));
    }
    MI_SYS_FlushInvCache(pstPreProcessedData->pdata, imageSize);

}

#define INNER_MOST_ALIGNMENT (8)
//#define ALIGN_UP(val, alignment) ((( (val)+(alignment)-1)/(alignment))*(alignment))


void ShowFloatOutPutTensor(float *pfBBox,float *pfClass, float *pfscore, float *pfDetect){
    // show bbox
    int s32DetectCount = round(*pfDetect);
    std::cout<<"BBox:"<<std::endl;
    std::cout.flags(std::ios::left);
    for(int i=0;i<s32DetectCount;i++)
    {
       for(int j=0;j<4;j++)
       {
            std::cout<<std::setw(15)<<*(pfBBox+(i*ALIGN_UP(4,INNER_MOST_ALIGNMENT))+j);
       }
       if (i!=0)
       {
            std::cout<<std::endl;
       }
    }

    //show class
    std::cout<<"Class:"<<std::endl;
    for(int  i=0;i<s32DetectCount;i++)
    {
       std::cout<<std::setw(15)<<round(*(pfClass+i));
    }
    std::cout<<std::endl;

    //show score
    std::cout<<"score:"<<std::endl;
    for(int  i=0;i<s32DetectCount;i++)
    {
       std::cout<<std::setw(15)<<*(pfscore+i);
    }
    std::cout<<std::endl;

    //show deteccout
    std::cout<<"DetectCount"<<std::endl;
    std::cout<<s32DetectCount<<std::endl;


}

 void WriteVisualizeBBox(std::string strImageName,
                   const std::vector<DetectionBBoxInfo > detections,
                   const float threshold, const std::vector<cv::Scalar>& colors,
                   const std::map<int, std::string>& classToDisName)
  {
  // Retrieve detections.
  cv::Mat image = cv::imread(strImageName, -1);
  std::map< int, std::vector<DetectionBBoxInfo> > detectionsInImage;

 for (unsigned int j = 0; j < detections.size(); j++) {
   DetectionBBoxInfo bbox;
   const int label = detections[j].classID;
   const float score = detections[j].score;
   if (score < threshold) {
     continue;
   }


   bbox.xmin =  detections[j].xmin*image.cols;
   bbox.xmin = bbox.xmin < 0 ? 0:bbox.xmin ;

   bbox.ymin =  detections[j].ymin*image.rows;
   bbox.ymin = bbox.ymin < 0 ? 0:bbox.ymin ;

   bbox.xmax =  detections[j].xmax*image.cols;
   bbox.xmax = bbox.xmax > image.cols?image.cols:bbox.xmax;

   bbox.ymax =  detections[j].ymax*image.rows;
   bbox.ymax = bbox.ymax > image.rows ? image.rows:bbox.ymax ;

   bbox.score = score;
   bbox.classID = label;
   detectionsInImage[label].push_back(bbox);
 }


  int fontface = cv::FONT_HERSHEY_SIMPLEX;
  double scale = 0.5;
  int thickness = 2;
  int baseline = 0;
  char buffer[50];

    // Show FPS.
//    snprintf(buffer, sizeof(buffer), "FPS: %.2f", fps);
//    cv::Size text = cv::getTextSize(buffer, fontface, scale, thickness,
//                                    &baseline);
//    cv::rectangle(image, cv::Point(0, 0),
//                  cv::Point(text.width, text.height + baseline),
//                  CV_RGB(255, 255, 255), CV_FILLED);
//    cv::putText(image, buffer, cv::Point(0, text.height + baseline / 2.),
//                fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
    // Draw bboxes.
    std::string name = strImageName;

    unsigned int pos = strImageName.rfind("/");
    if (pos > 0 && pos < strImageName.size()) {
      name = name.substr(pos + 1);
    }

    std::string strOutImageName = name;
    strOutImageName = "out_"+strOutImageName;

    pos = name.rfind(".");
    if (pos > 0 && pos < name.size()) {
      name = name.substr(0, pos);
    }

    name = name + ".txt";
    std::ofstream file(name);
    for (std::map<int, std::vector<DetectionBBoxInfo> >::iterator it =
         detectionsInImage.begin(); it != detectionsInImage.end(); ++it) {
      int label = it->first;
      std::string label_name = "Unknown";
      if (classToDisName.find(label) != classToDisName.end()) {
        label_name = classToDisName.find(label)->second;
      }
      const cv::Scalar& color = colors[label];
      const std::vector<DetectionBBoxInfo>& bboxes = it->second;
      for (unsigned int j = 0; j < bboxes.size(); ++j) {
        cv::Point top_left_pt(bboxes[j].xmin, bboxes[j].ymin);
        cv::Point bottom_right_pt(bboxes[j].xmax, bboxes[j].ymax);
        cv::rectangle(image, top_left_pt, bottom_right_pt, color, 1);
        cv::Point bottom_left_pt(bboxes[j].xmin, bboxes[j].ymax);
        snprintf(buffer, sizeof(buffer), "%s: %.2f", label_name.c_str(),
                 bboxes[j].score);
        cv::Size text = cv::getTextSize(buffer, fontface, scale, thickness,
                                        &baseline);
        cv::rectangle(
            image, top_left_pt + cv::Point(0, 0),
            top_left_pt + cv::Point(text.width, -text.height - baseline),
            color, 1);
        cv::putText(image, buffer, top_left_pt - cv::Point(0, baseline),
                    fontface, scale, CV_RGB(0,255, 0), thickness, 8);
        file << label_name << " " << bboxes[j].score << " "
            << bboxes[j].xmin / image.cols << " "
            << bboxes[j].ymin / image.rows << " "
            << bboxes[j].xmax / image.cols
            << " " << bboxes[j].ymax / image.rows << std::endl;
      }
    }
    file.close();
    cv::imwrite(strOutImageName.c_str(), image);

}

std::vector<DetectionBBoxInfo >  GetDetections(float *pfBBox,float *pfClass, float *pfScore, float *pfDetect)
{
    // show bbox
    int s32DetectCount = round(*pfDetect);
    std::vector<DetectionBBoxInfo > detections(s32DetectCount);
    for(int i=0;i<s32DetectCount;i++)
    {
        DetectionBBoxInfo  detection;
        memset(&detection,0,sizeof(DetectionBBoxInfo));
        //box coordinate
        detection.ymin =  *(pfBBox+(i*ALIGN_UP(4,INNER_MOST_ALIGNMENT))+0);
        detection.xmin =  *(pfBBox+(i*ALIGN_UP(4,INNER_MOST_ALIGNMENT))+1);
        detection.ymax =  *(pfBBox+(i*ALIGN_UP(4,INNER_MOST_ALIGNMENT))+2);
        detection.xmax =  *(pfBBox+(i*ALIGN_UP(4,INNER_MOST_ALIGNMENT))+3);


        //box class
        detection.classID = round(*(pfClass+i));


        //score
        detection.score = *(pfScore+i);
        detections.push_back(detection);

    }

    return detections;

}

int  GetLabels(const char *pLabelPath, char label[][LABEL_NAME_MAX_SIZE])
{
    std::ifstream LabelFile;
    LabelFile.open(pLabelPath);
    int n=0;
    while(!LabelFile.eof())
    {
        LabelFile.getline(&label[n][0],60);
        LabelFile.peek();
        n++;
        if(n>19){
                 break;
        }
        if(n>=LABEL_CLASS_COUNT)
        {
            std::cout<<"the labels have line:"<<n<<" ,it supass the available label array"<<std::endl;
        }
    }

    LabelFile.close();
    return n;

}
cv::Scalar HSV2RGB(const float h, const float s, const float v) {
  const int h_i = static_cast<int>(h * 6);
  const float f = h * 6 - h_i;
  const float p = v * (1 - s);
  const float q = v * (1 - f*s);
  const float t = v * (1 - (1 - f) * s);
  float r, g, b;
  switch (h_i) {
    case 0:
      r = v; g = t; b = p;
      break;
    case 1:
      r = q; g = v; b = p;
      break;
    case 2:
      r = p; g = v; b = t;
      break;
    case 3:
      r = p; g = q; b = v;
      break;
    case 4:
      r = t; g = p; b = v;
      break;
    case 5:
      r = v; g = p; b = q;
      break;
    default:
      r = 1; g = 1; b = 1;
      break;
  }
  return cv::Scalar(r * 255, g * 255, b * 255);
}
std::vector<cv::Scalar> GetColors(const int n)
{
  std::vector<cv::Scalar> colors;
  cv::RNG rng(12345);
  const float golden_ratio_conjugate = 0.618033988749895;
  const float s = 0.3;
  const float v = 0.99;
  for (int i = 0; i < n; ++i) {
    const float h = std::fmod(rng.uniform(0.f, 1.f) + golden_ratio_conjugate,
                              1.f);
    colors.push_back(HSV2RGB(h, s, v));
  }
  return colors;
}

static MI_BOOL GetTopN(float aData[], int dataSize, int aResult[], int TopN)
{
    int i, j, k;
    float data = 0;
    MI_BOOL bSkip = FALSE;

    for (i=0; i < TopN; i++)
    {
        data = -0.1f;
        for (j = 0; j < dataSize; j++)
        {
            if (aData[j] > data)
            {
                bSkip = FALSE;
                for (k = 0; k < i; k++)
                {
                    if (aResult[k] == j)
                    {
                        bSkip = TRUE;
                    }
                }

                if (bSkip == FALSE)
                {
                    aResult[i] = j;
                    data = aData[j];
                }
            }
        }
    }

    return TRUE;
}

bool has_suffix(const std::string& s, const std::string& suffix)
{
    return (s.size() >= suffix.size()) && equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

static void parse_images_dir(const std::string& base_path, std::vector<std::string>& file_path)
{
    DIR* dir;
    struct dirent* ptr;
    std::string base_path_str {base_path};
    if ((dir = opendir(base_path_str.c_str())) == NULL)
    {
        file_path.push_back(base_path_str);
        return;
    }
    if (base_path_str.back() != '/') {
        base_path_str.append("/");
    }

    while ((ptr = readdir(dir)) != NULL)
    {
        //std::cout<<"name - :"<<ptr->d_name<<std::endl;
        //if (ptr->d_type == DT_REG) {
            //std::cout<<"name_reg - :"<<ptr->d_name<<std::endl;
        if(strcmp(".", ptr->d_name) == 0)
            continue;
        if(has_suffix(ptr->d_name, ".jpg") || has_suffix(ptr->d_name, ".bmp") || has_suffix(ptr->d_name, ".png") || has_suffix(ptr->d_name, ".dmp")){
            std::string path = base_path_str + ptr->d_name;
            file_path.push_back(path);
        }
        //}
    }
    sort(file_path.begin(), file_path.end());
    closedir(dir);
}


int PWM_request(int num)
{
    int fd = -1;
    char buf[6];
    memset(buf,'\0',sizeof(buf));
    sprintf(buf, "%d", num);

    fd = open("/sys/class/pwm/pwmchip0/export", O_WRONLY);

    if(fd == -1)
    {
        printf("fail to open /sys/class/pwm/pwmchip0/export!\n");
        return -1;
    }

    if(write(fd, buf, sizeof(buf))<0)
    {
        printf("fail to write /sys/class/pwm/pwmchip0/export!\n");
        return -1;
    }
    close(fd);
    return 0;
}

int PWM_set_period(int num, int period)
{
    int fd = -1;
    char path[300];
    char buf[6];
    memset(buf,'\0',sizeof(buf));
    memset(path,'\0',sizeof(path));
    sprintf(path, "/sys/class/pwm/pwmchip0/pwm%d/period", num);
    fd = open(path, O_WRONLY);

    if(fd == -1)
    {
        // fail to open watchdog device
        printf("fail to open %s!\n",path);
        return -1;
    }
    sprintf(buf, "%d", period);
    if(write(fd, buf, sizeof(buf))<0)
    {
        printf("fail to write %s to %s!\n",buf,path);
        return -1;
    }
    close(fd);
    return 0;
}

int PWM_set_duty_cycle(int num, int duty_cycle)
{
    int fd = -1;
    char path[300];
    char buf[6];
    memset(buf,'\0',sizeof(buf));
    memset(path,'\0',sizeof(path));
    sprintf(path, "/sys/class/pwm/pwmchip0/pwm%d/duty_cycle", num);
    fd = open(path, O_WRONLY);

    if(fd == -1)
    {
        // fail to open watchdog device
        printf("fail to open %s!\n",path);
        return -1;
    }
    sprintf(buf, "%d", duty_cycle);
    if(write(fd, buf, sizeof(buf))<0)
    {
        printf("fail to write %s to %s!\n",buf,path);
        return -1;
    }
    close(fd);
    return 0;
}

int PWM_Enable(int num,int flag)
{
    int fd = -1;
    char path[300];
    char buf[6];
    memset(buf,'\0',sizeof(buf));
    memset(path,'\0',sizeof(path));
    sprintf(path, "/sys/class/pwm/pwmchip0/pwm%d/enable", num);
    fd = open(path, O_WRONLY);
    if(fd == -1)
    {
        // fail to open watchdog device
        printf("fail to open %s!\n",path);
        return -1;
    }
    sprintf(buf, "%d", flag);
    if(write(fd, buf, sizeof(buf))<0)
    {
        printf("fail to write %s to %s!\n",buf,path);
        return -1;
    }
    close(fd);
    return 0;
}


int getTypeSYze(MI_IPU_ELEMENT_FORMAT format){
    switch (format) {
        case MI_IPU_FORMAT_INT16:
            return sizeof(short);
        case MI_IPU_FORMAT_INT32:
            return sizeof(int);
        case MI_IPU_FORMAT_INT8:
            return sizeof(char);
        case MI_IPU_FORMAT_FP32:
            return sizeof(float);
        case MI_IPU_FORMAT_UNKNOWN:
        default:
            return 1;
    }
}

// cv::putText(input_image, label, cv::Point(left, top + (int)(label_size.height*scale)), FONT_FACE, scale, getColor(prc), THICKNESS);

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

void trackToImage(cv::Mat &inputImg, std::vector<STrack> &stracks, const std::vector<std::string> &labels, BYTETracker &tracker){
    int h = 0;
    for (int i = 0; i < stracks.size(); i++){
		vector<float> tlwh = stracks[i].tlwh;
		//bool vertical = tlwh[2] / tlwh[3] > 1.6;
		//if (tlwh[2] * tlwh[3] > 20 && !vertical)
		//{
		Scalar s = tracker.get_color(stracks[i].track_id);
// class - %s", labels[stracks[i].classId].c_str()
        h = tlwh[1]+1;
        string label = cv::format("obj - %d", stracks[i].track_id);
        cv::Size _size = draw_label(inputImg, label, tlwh[0], h, tlwh[2], stracks[i].score);//- (int)((float)height*(_confidence[idx])) 
        label = cv::format("score-%.2f", stracks[i].score);
        h += (_size.height+1);
        _size = draw_label(inputImg, label, tlwh[0], h, tlwh[2], stracks[i].score);
        label = cv::format("%s", labels[stracks[i].startClassId].c_str());
        h += (_size.height+1);
        _size = draw_label(inputImg, label, tlwh[0], h, tlwh[2], stracks[i].score);

        label = cv::format("state-%d", stracks[i].state);
        h += (_size.height+1);
        _size = draw_label(inputImg, label, tlwh[0], h, tlwh[2], stracks[i].score);

        label = cv::format("Vx-%.2f", stracks[i].mean(4));
        h += (_size.height+1);
        _size = draw_label(inputImg, label, tlwh[0], h, tlwh[2], stracks[i].score);
        label = cv::format("Vy-%.2f", stracks[i].mean(5));
        h += (_size.height+1);
        _size = draw_label(inputImg, label, tlwh[0], h, tlwh[2], stracks[i].score);
        label = cv::format("Va-%.2f", stracks[i].mean(6));
        h += (_size.height+1);
        _size = draw_label(inputImg, label, tlwh[0], h, tlwh[2], stracks[i].score);
        label = cv::format("Vh-%.2f", stracks[i].mean(7));
        h += (_size.height+1);
        _size = draw_label(inputImg, label, tlwh[0], h, tlwh[2], stracks[i].score);


        //int fontFace = cv::FONT_HERSHEY_DUPLEX, fontScale = _label.size() / 10;
        //cv::Size textSize = getTextSize(inputImg, fontFace, fontScale, 0, 0);
    	//putText(inputImg, _label, Point(tlwh[0], tlwh[1] - 5), fontFace, fontScale, Scalar(0, 255, 0), 2, LINE_AA);
        rectangle(inputImg, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), getColor(stracks[i].score), 1);

        cv::drawMarker(inputImg, Point2f(stracks[i].mean(0), stracks[i].mean(1)), Scalar(255, 0, 0), 0, 20, 1);
        cv::drawMarker(inputImg, Point2f(stracks[i].mean_prev(0), stracks[i].mean_prev(1)), Scalar(0, 0, 255), 0, 20, 1);

		//}
	} 
}

void trackTolog(std::ofstream &jfile, STrack &track){
    jfile << "\t{\n" 
        << "\t\t\"track_id\": " << track.track_id << ",\n"
        << "\t\t\"start_category_id\": " << track.startClassId << ",\n"
        << "\t\t\"bbox\": [\n"
            << "\t\t\t" << track.tlwh[0]<< ",\n"
            << "\t\t\t" << track.tlwh[1] << ",\n"
            << "\t\t\t" << track.tlwh[2] << ",\n"
            << "\t\t\t" << track.tlwh[3] << ",\n"
        << "\t\t],\n"
        << "\t\t\"start frame\": " << track.start_frame << ",\n"
        << "\t\t\"current frame\": " << track.frame_id << ",\n"
        << "\t\t\"score\": " << track.score << ",\n"
        << "\t\t\"Vx\": " << track.mean(4) << ",\n"
        << "\t\t\"Vy\": " << track.mean(5) << ",\n"
        << "\t\t\"Va\": " << track.mean(6) << ",\n"
        << "\t\t\"Vh\": " << track.mean(7) << ",\n"
        << "\t\t\"tracklet_len\": " << track.tracklet_len << ",\n"
        << "\t\t\"is_activated\": " << track.is_activated << ",\n"
        << "\t\t\"state\": " << track.state << ",\n"
        << "\t\t\"category_id\": " << track.ClassId << "\n"
        << "\t}";
    //<< std::endl;
}

void objTolog(std::ofstream &jfile, Object &object){
    jfile << "\t{\n" 
        << "\t\t\"score\": " << object.prob << ",\n"
        << "\t\t\"category_id\": " << object.label << ",\n"
        << "\t\t\"bbox\": [\n"
            << "\t\t\t" << object.rect.x << ",\n"
            << "\t\t\t" << object.rect.y << ",\n"
            << "\t\t\t" << object.rect.width << ",\n"
            << "\t\t\t" << object.rect.height << ",\n"
        << "\t\t]\n"
        << "\t}";
    //<< std::endl;
}
