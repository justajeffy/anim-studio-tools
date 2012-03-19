/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/Shader.h $"
 * SVN_META_ID = "$Id: Shader.h 30989 2010-05-11 23:23:13Z chris.cooper $"
 */

#ifndef bee_Shader_h
#define bee_Shader_h
#pragma once

#include <GL/glew.h>
#include <GL/glut.h>
#include "../kernel/string.h"
#include "../kernel/types.h"

namespace bee
{
	//-------------------------------------------------------------------------------------------------
	//! Shader is a GL utility class encapsulating a vertex <b>or</b> a fragment shader (GLSL)
	class Shader
	{
	public:
		//! Shader Type
		enum Type
		{
			eVertex = 0,
			eFragment,
			eGeometry, // not yet supported...
		};

		//! Constructor (needs the shader type)
		Shader( GLenum s_ShaderType );
		//! Destructor
		virtual ~Shader();

		//! Load the specified file
		bool load( String a_FileName );
		//! Compile it (using eventually an include code)
		bool compile( const char * a_IncludeCode = NULL );

		//! Get some info about the GL Program
		void getInfoLog( String & o_InfoLog );

		//! Returns GL ID
		inline UInt getID()
		{
			return m_ID;
		}
		//! Returns Shader Type
		inline GLenum getType()
		{
			return m_Type;
		}

		static char* fileRead( String a_FileName );

	private:
		friend class Program;
		void deleteCode();

	protected:
		GLenum m_Type; // GL_VERTEX_SHADER or GL_FRAGMENT_SHADER
		UInt m_ID;
		String m_FileName;
		char * m_Code;
	};
}
#endif // bee_Shader_h


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
