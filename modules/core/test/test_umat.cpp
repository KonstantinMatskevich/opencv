/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

using namespace cvtest;
using namespace testing;
using namespace cv;

#define EXPECT_MAT_NEAR(mat1, mat2, eps) \
{ \
   ASSERT_EQ(mat1.type(), mat2.type()); \
   ASSERT_EQ(mat1.size(), mat2.size()); \
   EXPECT_LE(cv::norm(mat1, mat2), eps); \
}\

PARAM_TEST_CASE(UMatBasicTests, int, int, Size, bool)
{
    Mat a, b, roi_a, roi_b;
    UMat ua, ub, roi_ua, roi_ub;
    int type;
    int depth;
    int cn;
    Size size;
    bool useRoi;
    Size roi_size;
    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        cn = GET_PARAM(1);
        size = GET_PARAM(2);
        useRoi = GET_PARAM(3);
        type = CV_MAKE_TYPE(depth, cn);
    }
};

CORE_TEST_P(UMatBasicTests, createUMat)
{
    int roi_shift_x = randomInt(0, size.width-1);
    int roi_shift_y = randomInt(0, size.height-1);
    roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
    a = randomMat(size, type, -100, 100);
    Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
    a.copyTo(ua);
    if(useRoi)
    {
        ua = UMat(ua, roi);
    }
    int dims = randomInt(2,CV_MAX_DIM);
    int _sz[CV_MAX_DIM];
    for( int i = 0; i<dims; i++)
    {
        _sz[i] = randomInt(1,100);
    }
    int *sz = _sz;
    int new_depth = randomInt(CV_8S, CV_64F);
    int new_cn = randomInt(1,4);
    ua.create(dims, sz, CV_MAKE_TYPE(new_depth, new_cn));

    for(int i = 0; i<dims; i++)
    {
        ASSERT_EQ(ua.size[i], sz[i]);
    }
    ASSERT_EQ(ua.dims, dims);
    ASSERT_EQ(ua.type(), CV_MAKE_TYPE(new_depth, new_cn) );
    Size new_size = randomSize(1, 1000);
    ua.create(new_size, CV_MAKE_TYPE(new_depth, new_cn) );
    ASSERT_EQ( ua.size(), new_size);
    ASSERT_EQ(ua.type(), CV_MAKE_TYPE(new_depth, new_cn) );
    ASSERT_EQ( ua.dims, 2);
}

CORE_TEST_P(UMatBasicTests, swap)
{
    a = randomMat(size,type, -100, 100);
    b = randomMat(size,type, -100, 100);
    a.copyTo(ua);
    b.copyTo(ub);
    if(useRoi)
    {
        int roi_shift_x = randomInt(0, size.width-1);
        int roi_shift_y = randomInt(0, size.height-1);
        roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
        Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
        roi_ua = UMat(ua,roi);
        roi_ub = UMat(ub,roi);
        UMat roi_uc = roi_ua, roi_ud = roi_ub;
        swap(roi_ua,roi_ub);
        EXPECT_MAT_NEAR(roi_uc, roi_ub, 0);
        EXPECT_MAT_NEAR(roi_ud, roi_ua, 0);
    }
    else
    {
        UMat uc = ua, ud = ub;
        swap(ua,ub);
        EXPECT_MAT_NEAR(ub,uc, 0);
        EXPECT_MAT_NEAR(ud, ua, 0);
    }
}

CORE_TEST_P(UMatBasicTests, base)
{
    a = randomMat(size,type, -100, 100);
    a.copyTo(ua);
    if(useRoi)
    {
        int roi_shift_x = randomInt(0, size.width-1);
        int roi_shift_y = randomInt(0, size.height-1);
        roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
        Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
        ua = UMat(ua,roi);
    }
    ub = ua.clone();
    EXPECT_MAT_NEAR(ub,ua,0);

    ASSERT_EQ(ua.channels(), cn);
    ASSERT_EQ(ua.depth(), depth);
    ASSERT_EQ(ua.type(), type);
    ASSERT_EQ(ua.elemSize(), a.elemSize());
    ASSERT_EQ(ua.elemSize1(), a.elemSize1());
    ASSERT_EQ(ub.empty(), ub.cols*ub.rows == 0);
    ub.release();
    ASSERT_TRUE( ub.empty() );
    if(useRoi && a.size() != ua.size())
    {
        ASSERT_EQ(ua.isSubmatrix(), true);
    }
    else
    {
        ASSERT_EQ(ua.isSubmatrix(), false);
    }

    int dims = randomInt(2,CV_MAX_DIM);
    int sz[CV_MAX_DIM];
    size_t total = 1;
    for(int i = 0; i<dims; i++)
    {
        sz[i] = randomInt(1,345);
        total *= (size_t)sz[i];
    }
    int new_type = CV_MAKE_TYPE(randomInt(CV_8S,CV_64F),randomInt(1,4));
    ub = UMat(dims, sz, new_type);
    ASSERT_EQ(ub.total(), total);
}

CORE_TEST_P(UMatBasicTests, copyTo)
{
    a = randomMat(size, type, -100, 100);
    a.copyTo(ua);
    if(useRoi)
    {
        int roi_shift_x = randomInt(0, size.width-1);
        int roi_shift_y = randomInt(0, size.height-1);
        roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
        Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
        roi_ua = UMat(ua, roi);
        roi_ua.copyTo(roi_ub);
        EXPECT_MAT_NEAR(roi_ua, roi_ub, 0);
        int i = randomInt(0, roi_ua.cols-1);
        roi_ua.col(i).copyTo(roi_ub);
        EXPECT_MAT_NEAR(roi_ua.col(i), roi_ub, 0);
        i = randomInt(0, roi_ua.rows-1);
        roi_ua.row(i).copyTo(roi_ub);
        EXPECT_MAT_NEAR(roi_ua.row(i), roi_ub, 0);
    }
    else
    {
        ua.copyTo(ub);
        EXPECT_MAT_NEAR(ua, ub, 0);
        int i = randomInt(0, ua.cols-1);
        ua.col(i).copyTo(ub);
        EXPECT_MAT_NEAR(ua.col(i), ub, 0);
        i = randomInt(0, a.rows-1);
        ua.row(i).copyTo(ub);
        EXPECT_MAT_NEAR(ua.row(i), ub, 0);
    }
}

CORE_TEST_P(UMatBasicTests, DISABLED_GetUMat)
{
    a = randomMat(size, type, -100, 100);
    a.copyTo(ua);
    if(useRoi)
    {
        int roi_shift_x = randomInt(0, size.width-1);
        int roi_shift_y = randomInt(0, size.height-1);
        roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
        Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
        roi_a = Mat(a, roi);
        roi_b = roi_a;
        EXPECT_MAT_NEAR(roi_a.getUMat(ACCESS_RW).getMat(ACCESS_RW), roi_b, 0);
        roi_ua = UMat(ua,roi);
        roi_ub = roi_ua;
        EXPECT_MAT_NEAR(roi_ua.getMat(ACCESS_RW).getUMat(ACCESS_RW), roi_ub, 0);
    }
    else
    {
        b = a;
        EXPECT_MAT_NEAR(a.getUMat(ACCESS_RW).getMat(ACCESS_RW), b, 0);
        ub=ua;
        EXPECT_MAT_NEAR(ua.getMat(ACCESS_RW).getUMat(ACCESS_RW), ub, 0);
    }
}

INSTANTIATE_TEST_CASE_P(Umat, UMatBasicTests, Combine(UMAT_TEST_DEPTH, UMAT_TEST_CHANNELS, UMAT_TEST_SIZES, Values(true,false) ) );
    //Values(CV_32S), Values(1), Values(cv::Size(11,11)), Values(true, false) ) );

PARAM_TEST_CASE(UMatTestReshape,  int, int, Size, bool)
{
    Mat a, roi_a;
    UMat ua, ub;
    int type;
    int depth;
    int cn;
    Size size;
    bool useRoi;
    Size roi_size;
    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        cn = GET_PARAM(1);
        size = GET_PARAM(2);
        useRoi = GET_PARAM(3);
        type = CV_MAKE_TYPE(depth, cn);
    }
};

CORE_TEST_P(UMatTestReshape, reshape)
{
    a = randomMat(size,type, -100, 100);
    a.copyTo(ua);
    if(useRoi)
    {
        int roi_shift_x = randomInt(0, size.width-1);
        int roi_shift_y = randomInt(0, size.height-1);
        roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
        Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
        ua = UMat(ua,roi);
    }

    int nChannels = randomInt(1,4);
    Mat tmp;

    if ((ua.cols*ua.rows*ua.channels())%nChannels != 0)
    {
        EXPECT_ANY_THROW( ub = ua.reshape(nChannels));
    }
    else
    {
        ub = ua.reshape(nChannels);
        ASSERT_EQ(ub.channels(),nChannels);
        ASSERT_EQ(ub.channels()*ub.cols*ub.rows, ua.channels()*ua.cols*ua.rows);

        ub.copyTo(tmp);
        if(useRoi)
        {
            EXPECT_MAT_NEAR(tmp.reshape(1,1), roi_a.reshape(1,1), 0);
        }
        else
        {
            EXPECT_MAT_NEAR(tmp.reshape(1,1), a.reshape(1,1), 0);
        }

        int new_rows = randomInt(1, INT_MAX);
        if ( (ua.cols*ua.rows*ua.channels())%new_rows != 0)
        {
            EXPECT_ANY_THROW (ub = ua.reshape(nChannels, new_rows) );
        }
        else
        {
            EXPECT_NO_THROW ( ub = ua.reshape(nChannels, new_rows) );
            ASSERT_EQ(ub.channels(),nChannels);
            ASSERT_EQ(ub.rows, new_rows);
            ASSERT_EQ(ub.channels()*ub.cols*ub.rows, ua.channels()*ua.cols*ua.rows);

            ub.copyTo(tmp);
            if(useRoi)
            {
                EXPECT_MAT_NEAR(tmp.reshape(1,1), roi_a.reshape(1,1), 0);
            }
            else
            {
                EXPECT_MAT_NEAR(tmp.reshape(1,1), a.reshape(1,1), 0);
            }

            cv::Size new_size = randomSize(0, INT_MAX);
            int sz[] = {size.height, size.width};
            if( (ua.cols*ua.rows*ua.channels()) % (size.height*size.width) != 0 )
            {
                EXPECT_ANY_THROW( ub = ua.reshape(nChannels, ua.dims, sz) );
            }
            else
            {
                EXPECT_NO_THROW ( ub = ua.reshape(nChannels, ua.dims, sz) );
                ASSERT_EQ(ub.channels(),nChannels);
                ASSERT_EQ(ub.rows, new_size.width);
                ASSERT_EQ(ub.cols, new_size.height);
                ASSERT_EQ(ub.channels()*ub.cols*ub.rows, ua.channels()*ua.cols*ua.rows);

                ub.copyTo(tmp);
                if(useRoi)
                {
                    EXPECT_MAT_NEAR(tmp.reshape(1,1), roi_a.reshape(1,1), 0);
                }
                else
                {
                    EXPECT_MAT_NEAR(tmp.reshape(1,1), a.reshape(1,1), 0);
                }
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(Umat, UMatTestReshape, Combine(UMAT_TEST_DEPTH, UMAT_TEST_CHANNELS, UMAT_TEST_SIZES, Values(true,false) ));
    //Values(CV_32S), Values(1), Values(cv::Size(11,11)), Values(true, false) ) );

PARAM_TEST_CASE(UMatTestRoi, int, int, Size)
{
    Mat a, b, roi_a, roi_b;
    UMat ua, ub, roi_ua, roi_ub;
    int type;
    int depth;
    int cn;
    Size size;
    Size roi_size;
    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        cn = GET_PARAM(1);
        size = GET_PARAM(2);
        type = CV_MAKE_TYPE(depth, cn);
    }
};

CORE_TEST_P(UMatTestRoi, createRoi)
{
    int roi_shift_x = randomInt(0, size.width-1);
    int roi_shift_y = randomInt(0, size.height-1);
    roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
    a = randomMat(size, type, -100, 100);
    Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
    roi_a = Mat(a, roi);
    a.copyTo(ua);
    roi_ua = UMat(ua, roi);
    roi_a.copyTo(roi_ub);
    roi_ua.copyTo(roi_b);
    EXPECT_MAT_NEAR(roi_a, roi_b, 0);
    EXPECT_MAT_NEAR(roi_ub, roi_ua, 0);
}

CORE_TEST_P(UMatTestRoi, locateRoi)
{
    int roi_shift_x = randomInt(0, size.width-1);
    int roi_shift_y = randomInt(0, size.height-1);
    roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
    a = randomMat(size, type, -100, 100);
    Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
    roi_a = Mat(a, roi);
    a.copyTo(ua);
    roi_ua = UMat(ua,roi);
    Size sz, usz;
    Point p, up;
    roi_a.locateROI(sz, p);
    roi_ua.locateROI(usz, up);
    ASSERT_EQ(sz, usz);
    ASSERT_EQ(p, up);
}

CORE_TEST_P(UMatTestRoi, adjustRoi)
{
    int roi_shift_x = randomInt(0, size.width-1);
    int roi_shift_y = randomInt(0, size.height-1);
    roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
    a = randomMat(size, type, -100, 100);
    Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
    a.copyTo(ua);
    roi_ua = UMat( ua, roi);
    int adjLeft = randomInt(-(roi_ua.cols/2), (size.width-1)/2);
    int adjRight = randomInt(-(roi_ua.cols/2), (size.width-1)/2);
    int adjTop = randomInt(-(roi_ua.rows/2), (size.height-1)/2);
    int adjBot = randomInt(-(roi_ua.rows/2), (size.height-1)/2);
    roi_ua.adjustROI(adjTop, adjBot, adjLeft, adjRight);
    roi_shift_x = max(0, roi.x-adjLeft);
    roi_shift_y = max(0, roi.y-adjTop);
    Rect new_roi( roi_shift_x, roi_shift_y, min(roi.width+adjRight+adjLeft, size.width-roi_shift_x), min(roi.height+adjBot+adjTop, size.height-roi_shift_y) );
    roi_ub = UMat(ua, new_roi);
    EXPECT_MAT_NEAR(roi_ua, roi_ub, 0);
}

INSTANTIATE_TEST_CASE_P(Umat, UMatTestRoi, Combine(UMAT_TEST_DEPTH, UMAT_TEST_CHANNELS, UMAT_TEST_SIZES ));
    //Values(CV_32S), Values(1), Values(cv::Size(11,11)) ) );

PARAM_TEST_CASE(UMatTestSizeOperations, int, int, Size, bool)
{
    Mat a, b, roi_a, roi_b;
    UMat ua, ub, roi_ua, roi_ub;
    int type;
    int depth;
    int cn;
    Size size;
    Size roi_size;
    bool useRoi;
    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        cn = GET_PARAM(1);
        size = GET_PARAM(2);
        useRoi = GET_PARAM(3);
        type = CV_MAKE_TYPE(depth, cn);
    }
};

CORE_TEST_P(UMatTestSizeOperations, copySize)
{
    Size s = randomSize(1,300);
    a = randomMat(size, type, -100, 100);
    b = randomMat(s, type, -100, 100);
    a.copyTo(ua);
    b.copyTo(ub);
    if(useRoi)
    {
        int roi_shift_x = randomInt(0, size.width-1);
        int roi_shift_y = randomInt(0, size.height-1);
        roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
        Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
        ua = UMat(ua,roi);

        roi_shift_x = randomInt(0, s.width-1);
        roi_shift_y = randomInt(0, s.height-1);
        roi_size = Size(s.width - roi_shift_x, s.height - roi_shift_y);
        roi = Rect(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
        ub = UMat(ub, roi);
    }
    ua.copySize(ub);
    ASSERT_EQ(ua.size, ub.size);
}

INSTANTIATE_TEST_CASE_P(Umat, UMatTestSizeOperations, Combine(UMAT_TEST_DEPTH, UMAT_TEST_CHANNELS, UMAT_TEST_SIZES, Values(true,false) ));
    //Values(CV_32S), Values(1), Values(cv::Size(11,11)), Values(true, false) ) );


PARAM_TEST_CASE(UMatTestUMatOperations, int, int, Size, bool)
{
    Mat a, b, roi_a, roi_b;
    UMat ua, ub, roi_ua, roi_ub;
    int type;
    int depth;
    int cn;
    Size size;
    Size roi_size;
    bool useRoi;
    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        cn = GET_PARAM(1);
        size = GET_PARAM(2);
        useRoi = GET_PARAM(3);
        type = CV_MAKE_TYPE(depth, cn);
    }
};

CORE_TEST_P(UMatTestUMatOperations, transpose) // UMatExpr is not implemented
{
    /*a = randomMat(size, type, -100, 100);
    if(useRoi)
    {
        int roi_shift_x = randomInt(0, size.width-1);
        int roi_shift_y = randomInt(0, size.height-1);
        roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
        Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
        a.copyTo(ua);
        roi_ua = UMat(ua,roi);
        roi_ub = roi_ua.t().t();
        EXPECT_MAT_NEAR(roi_ua, roi_ub, 0);
        roi_ub = roi_ua.t();
        ASSERT_EQ(roi_ub.rows, roi_ua.cols);
        ASSERT_EQ(roi_ub.cols, roi_ua.rows);
    }
    else
    {
        a.copyTo(ua);
        ub = ua.t().t();
        EXPECT_MAT_NEAR(ua, ub, 0);
        ub = ua.t();
        ASSERT_EQ(ub.rows, ua.cols);
        ASSERT_EQ(ub.cols, ua.rows);
    }*/
}

CORE_TEST_P(UMatTestUMatOperations, diag)
{
    a = randomMat(size, type, -100, 100);
    Mat new_diag;
    if(useRoi)
    {
        int roi_shift_x = randomInt(0, size.width-1);
        int roi_shift_y = randomInt(0, size.height-1);
        roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
        Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
        a.copyTo(ua);
        roi_ua = UMat(ua,roi);
        roi_a = Mat(a, roi);
        int n = randomInt(0, roi_ua.cols-1);
        roi_ub = roi_ua.diag(n);
        roi_b = roi_a.diag(n);
        roi_ub.copyTo(roi_a);
        EXPECT_MAT_NEAR(roi_a,roi_b, 0);
        roi_b.copyTo(roi_ua);
        EXPECT_MAT_NEAR(roi_ua, roi_ub, 0);
        new_diag = randomMat(Size(roi_ua.rows, 1), type, -100, 100);
        new_diag.copyTo(roi_ub);
        roi_ua = cv::UMat::diag(roi_ub); //it fails because function UMat& UMat::operator = (const Scalar&) is not impemented
        EXPECT_MAT_NEAR(roi_ua.diag(), roi_ub, 0);
    }
    else
    {
        a.copyTo(ua);
        int n = randomInt(0, ua.cols-1);
        ub = ua.diag(n);
        b = a.diag(n);
        ub.copyTo(a);
        EXPECT_MAT_NEAR(a,b, 0);
        b.copyTo(ua);
        EXPECT_MAT_NEAR(ua,ub, 0);
        new_diag = randomMat(Size(ua.rows, 1), type, -100, 100);
        new_diag.copyTo(ub);
        ua = cv::UMat::diag(ub); //it fails because function UMat& UMat::operator = (const Scalar&) is not impemented
        EXPECT_MAT_NEAR(ua.diag(), ub, 0);
    }
}

CORE_TEST_P(UMatTestUMatOperations, dotUMat)
{
    a = randomMat(size, type, -100, 100);
    b = randomMat(size, type, -100, 100);
    a.copyTo(ua);
    b.copyTo(ub);
    //ASSERT_EQ(ua.dot(ub), a.dot(b));
}

INSTANTIATE_TEST_CASE_P(Umat, UMatTestUMatOperations, Combine(UMAT_TEST_DEPTH, UMAT_TEST_CHANNELS, UMAT_TEST_SIZES, Values(true,false) ));
    //Values(CV_32S), Values(1), Values(cv::Size(11,11)), Values(true, false) ) );
