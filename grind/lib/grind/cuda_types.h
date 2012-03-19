/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: cuda_types.h 42544 2010-08-17 04:31:03Z allan.johns $"
 */

#ifndef grind_cuda_types_h
#define grind_cuda_types_h

#include <OpenEXR/ImathVec.h>
#include <OpenEXR/half.h>

//-------------------------------------------------------------------------------------------------
// central place for setting grid dimensions for cuda kernel eval
#define GRID_DIM_X 128
#define GRID_DIM_Y 1
#define GRID_DIM_Z 1

namespace Imath {
	typedef Vec3<half> V3h;
	typedef Vec4<half> V4h;
}

#endif /* grind_cuda_types_h */


/***
    Copyright 2008-2012 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios)

    This file is part of anim-studio-tools.

    anim-studio-tools is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    anim-studio-tools is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with anim-studio-tools.  If not, see <http://www.gnu.org/licenses/>.
***/
