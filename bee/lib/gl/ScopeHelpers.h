/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn.drd.int/drd/apps/bee/trunk/lib/gl/glExtensions.h $"
 * SVN_META_ID = "$Id: ScopeHelpers.h 23077 2010-02-18 03:57:01Z chris.cooper $"
 */

#ifndef GL_SCOPEHELPERS_H_
#define GL_SCOPEHELPERS_H_

namespace bee
{

	//! scoped state handling for glDepthMask
	struct glDepthMaskHelper
	{
		GLboolean m_OriginalState;

		glDepthMaskHelper( GLboolean desired_state )
		{
			glGetBooleanv( GL_DEPTH_WRITEMASK, &m_OriginalState );
			glDepthMask( desired_state );
		}

		~glDepthMaskHelper()
		{
			glDepthMask( m_OriginalState );
		}
	};

	//! scoped state handling for glDepthMask
	struct glEnableHelper
	{
		GLboolean m_OriginalState;
		GLenum m_Flag;

		glEnableHelper( GLenum flag ) :
			m_Flag( flag )
		{
			glGetBooleanv( flag, &m_OriginalState );
			glEnable( m_Flag );
		}

		~glEnableHelper()
		{
			if ( m_OriginalState ) glEnable( m_Flag );
			else glDisable( m_Flag );
		}
	};

	//! scoped state handling for glLineWidth
	struct glLineWidthHelper
	{
		GLfloat m_OriginalState;
		GLenum m_Flag;

		glLineWidthHelper( GLfloat desired_value )
		{
			glGetFloatv( GL_LINE_WIDTH, &m_OriginalState );
			glLineWidth( desired_value );
		}

		~glLineWidthHelper()
		{
			glLineWidth( m_OriginalState );
		}
	};

	//! scoped state handling for glBlendFunc
	struct glBlendFuncHelper
	{
		GLint m_src;
		GLint m_dst;

		glBlendFuncHelper( 	GLenum src,
							GLenum dst )
		{
			glGetIntegerv( GL_BLEND_SRC, &m_src );
			glGetIntegerv( GL_BLEND_DST, &m_dst );
			glBlendFunc( src, dst );
		}

		~glBlendFuncHelper()
		{
			glBlendFunc( m_src, m_dst );
		}
	};
}

#endif /* GLEXTENSIONS_H_ */


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
