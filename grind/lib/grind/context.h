/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: context.h 42544 2010-08-17 04:31:03Z allan.johns $"
 */

#ifndef grind_context_h
#define grind_context_h

//-------------------------------------------------------------------------------------------------
#include <OpenEXR/ImathVec.h>

namespace grind {

//-------------------------------------------------------------------------------------------------
//! a class for detecting context that grind is operating in
struct ContextInfo
{
	//! singleton access
	static ContextInfo& instance();

	//! construct and detect
	ContextInfo();

	//! does the current context have a gpu?
	bool hasGPU() const;

	//! is grind in emulation mode?
	bool hasEmulation() const;

	//! does the current context have OpenGL?
	bool hasOpenGL() const;

	//! does the current context have renderman RX?
	bool hasRX() const;

	//! @cond DEV

	//! dump info about the context
	void dump() const;

	//! get the current eye position
	Imath::V3f eyePos() const;

	//! get the fraction of gpu mem available (ie between zero and one)
	float getGpuMemAvailable( int a_Gpu = 0 );

	//! get the fraction of gpu mem available (ie between zero and one)
	unsigned int getFreeMem( int a_Gpu = 0 );

	//! get the fraction of gpu mem available (ie between zero and one)
	unsigned int getTotalMem( int a_Gpu = 0 );

	~ContextInfo();

private:

	bool m_HasGPU;
	bool m_HasRX;

	unsigned int m_ThreadId;
	//! @endcond
};

} // grind

#endif /* grind_context_h */


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
