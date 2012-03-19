/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/PBO.h $"
 * SVN_META_ID = "$Id: PBO.h 30989 2010-05-11 23:23:13Z chris.cooper $"
 */

#ifndef bee_PBO_h
#define bee_PBO_h
#pragma once

#include <GL/glew.h>
#include <GL/glut.h>
#include "../kernel/types.h"
#include "Texture.h"

namespace bee
{
	class PBO
	{
	public:
		PBO( UInt a_Width, UInt a_Height, const void * a_Buffer, UInt a_BufferSize );
		~PBO();

		void use() const;
		void release() const;

		UInt getWidth() const { return m_Width; }
		UInt getHeight() const { return m_Height; }
		const void * getBuffer() const { return m_Buffer; }
		UInt getBufferSize() const { return m_BufferSize; }

	private:

		UInt m_BufGLId;

		UInt m_Width;
		UInt m_Height;
		const void * m_Buffer;
		UInt m_BufferSize;
	};
}
#endif // bee_PBO_h


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
