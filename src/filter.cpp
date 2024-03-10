/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>
#include "nppi_data_exchange_and_initialization.h"

bool printfNPPinfo(int argc, char *argv[])
{
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}

int main(int argc, char *argv[])
{
  printf("%s Starting...\n\n", argv[0]);

  try
  {
    std::string sFilename;
    char *filePath;

    findCudaDevice(argc, (const char **)argv);

    if (printfNPPinfo(argc, argv) == false)
    {
      exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input"))
    {
      getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
    }
    else
    {
      filePath = sdkFindFilePath("data/Lena.pgm", argv[0]);
    }

    if (filePath)
    {
      sFilename = filePath;
    }
    else
    {
      sFilename = "data/Lena.pgm";
    }

    // if we specify the filename at the command line, then we only test
    // sFilename[0].
    int file_errors = 0;
    std::ifstream infile(sFilename.data(), std::ifstream::in);

    if (infile.good())
    {
      std::cout << "filter opened: <" << sFilename.data()
                << "> successfully!" << std::endl;
      file_errors = 0;
      infile.close();
    }
    else
    {
      std::cout << "filter unable to open: <" << sFilename.data() << ">"
                << std::endl;
      file_errors++;
      infile.close();
    }

    if (file_errors > 0)
    {
      exit(EXIT_FAILURE);
    }

    std::string sResultFilename = sFilename;

    std::string::size_type dot = sResultFilename.rfind('.');

    if (dot != std::string::npos)
    {
      sResultFilename = sResultFilename.substr(0, dot);
    }

    std::string sResultFilename2 = sResultFilename;

    sResultFilename += "_filter.pgm";
    sResultFilename2 += "_filterdist.pgm";

    if (checkCmdLineFlag(argc, (const char **)argv, "output"))
    {
      char *outputFilePath;
      getCmdLineArgumentString(argc, (const char **)argv, "output",
                               &outputFilePath);
      sResultFilename = outputFilePath;
    }

    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load gray-scale image from disk
    npp::loadImage(sFilename, oHostSrc);
    // declare a device image and copy construct from the host image,
    // i.e. upload host to device
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

    NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    NppiPoint oSrcOffset = {0, 0};

    // create struct with ROI size
    NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

    // allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
    npp::ImageNPP_16u_C1 oDeviceDst2(oSizeROI.width, oSizeROI.height);   

    int hpBufferSize;
    NPP_CHECK_NPP(nppiFilterCannyBorderGetBufferSize(oSizeROI, &hpBufferSize));

    //Allocate scratch memory for canny detection
    Npp8u *pDeviceBuffer = nullptr;
    cudaMalloc((void **)&pDeviceBuffer, hpBufferSize);

    // run sharpen filter
    /* NPP_CHECK_NPP(nppiFilterSharpenBorder_8u_C1R(
        oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
        oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI,
        NPP_BORDER_REPLICATE));
    */
    Npp16s nLow = 72;
    Npp16s nHigh = 250;

    //Run edge detection
    NPP_CHECK_NPP(nppiFilterCannyBorder_8u_C1R(
        oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
        oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI,
        NPP_FILTER_SOBEL, //NPP_FILTER_SCHARR
        NPP_MASK_SIZE_3_X_3,
        nLow, nHigh,
        nppiNormL2, //nppiNormL1, nppiNormL2,nppiNormInf
        NPP_BORDER_REPLICATE,
        pDeviceBuffer));

    //Clear scratch memory for canny detection
    cudaFree(pDeviceBuffer);

    //Get size of scratch buffer for Distance Transform
    size_t nScratchBufferSize;
    NPP_CHECK_NPP(nppiDistanceTransformPBAGetBufferSize(oSizeROI, &nScratchBufferSize));

    // Allocate scratch buffer
    Npp8u *pScratchDeviceBuffer;
    cudaMalloc((void **)&pScratchDeviceBuffer, nScratchBufferSize);

    //Configure stream
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    cudaError_t cudaError = cudaGetDevice(&nppStreamCtx.nCudaDeviceId);
    if (cudaError != cudaSuccess)
    {
      printf("CUDA error: no devices supporting CUDA.\n");
      return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;
    }

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("CUDA Driver  Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("CUDA Runtime Version: %d.%d\n\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    cudaError = cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMajor,
                                       cudaDevAttrComputeCapabilityMajor,
                                       nppStreamCtx.nCudaDeviceId);
    if (cudaError != cudaSuccess)
      return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;

    cudaError = cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMinor,
                                       cudaDevAttrComputeCapabilityMinor,
                                       nppStreamCtx.nCudaDeviceId);
    if (cudaError != cudaSuccess)
      return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;

    cudaError = cudaStreamGetFlags(nppStreamCtx.hStream, &nppStreamCtx.nStreamFlags);

    cudaDeviceProp oDeviceProperties;

    cudaError = cudaGetDeviceProperties(&oDeviceProperties, nppStreamCtx.nCudaDeviceId);

    nppStreamCtx.nMultiProcessorCount = oDeviceProperties.multiProcessorCount;
    nppStreamCtx.nMaxThreadsPerMultiProcessor = oDeviceProperties.maxThreadsPerMultiProcessor;
    nppStreamCtx.nMaxThreadsPerBlock = oDeviceProperties.maxThreadsPerBlock;
    nppStreamCtx.nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock;
    
    //Set the min/max to detect the sites
    Npp8u nMinSiteValue = 255;
    Npp8u nMaxSiteValue = 255;

    //Run euclidean distance transform
    NPP_CHECK_NPP(nppiDistanceTransformPBA_8u16u_C1R_Ctx(oDeviceDst.data(), oDeviceDst.pitch(), nMinSiteValue, nMaxSiteValue,
                                                         0, 0,
                                                         0, 0,
                                                         0, 0,
                                                         oDeviceDst2.data(), oDeviceDst2.pitch(),
                                                         oSizeROI, pScratchDeviceBuffer, nppStreamCtx));

    //Clear scratch memory
    cudaFree(pScratchDeviceBuffer);

    //Convert image from 16 bits to 8 bits
    npp::ImageNPP_8u_C1 oDeviceDst3(oSizeROI.width, oSizeROI.height);
    NPP_CHECK_NPP(nppiConvert_16u8u_C1R(oDeviceDst2.data(), oDeviceDst2.pitch(), oDeviceDst3.data(), oDeviceDst3.pitch(), oSizeROI));

    // declare a host image for the result
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    //npp::ImageCPU_16u_C1 oHostDst2(oDeviceDst2.size());

    npp::ImageCPU_8u_C1 oHostDst3(oDeviceDst3.size());
    // and copy the device result data into it
    //oDeviceDst2.copyTo(oHostDst2.data(), oHostDst2.pitch());
    oDeviceDst3.copyTo(oHostDst3.data(), oHostDst3.pitch());

    saveImage(sResultFilename, oHostDst);
    //saveImage(sResultFilename2, oHostDst2);
    saveImage(sResultFilename2, oHostDst3);
    std::cout << "Saved images: " << sResultFilename << " and " << sResultFilename2 << std::endl;

    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());
    nppiFree(oDeviceDst2.data());
    nppiFree(oDeviceDst3.data());

    exit(EXIT_SUCCESS);
  }
  catch (npp::Exception &rException)
  {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  }
  catch (...)
  {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}
