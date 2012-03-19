/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/Program.h $"
 * SVN_META_ID = "$Id: Program.h 59286 2010-12-16 03:42:24Z stephane.bertout $"
 */

#ifndef bee_Program_h
#define bee_Program_h


#include <GL/glew.h>
#include <GL/glut.h>
#include "../kernel/string.h"
#include "../kernel/types.h"
#include "../math/Imath.h"

//#include <boost/scoped_ptr.hpp>
#include "../kernel/smartPointers.h"

namespace bee
{
	class Shader;

	//-------------------------------------------------------------------------------------------------
	//! Program is a GL utility class encapsulating a vertex and a fragment shader (supports GLSL)
	class Program
	{
	public:
		//! Constructor taking the filenames of the shaders to load and associate with this Program
		Program( 	const char* a_VertexShaderName,
					const char* a_FragmentShaderName,
					bool a_UseIncludeFile = true );

		//! Constructor taking the filenames of the shaders to load and associate with this Program
		Program( 	const char* a_VertexShaderName,
					const char* a_FragmentShaderName,
					const char* a_GeometryShaderName,
					unsigned int a_GeomInType,
					unsigned int a_GeomOutType,
					bool a_UseIncludeFile = true );

		//! Destructor
		virtual ~Program();

		//! Setup GL State
		void use() const;

		//! Reset GL State;
		void release() const;

		//! Specify the filename of a GLSL utility file you would like to include with ALL the shaders loaded (linker will strip unused methods)
		static void useIncludeFile( const char * a_IncludeShaderFile );

		//! Specify a GLSL utility string you would like to include with ALL the shaders loaded (linker will strip unused methods)
		static void useIncludeString( const char * a_IncludeShaderString );

		//! Get some info about the GL Program
		void getInfoLog( String & o_InfoLog );

		//! Returns location of specified uniform (-1 if not found)
		Int getUniformLocation( const char * name ) const;

		//! Set int type uniform
		void setUniform( 	const char* name,
							int val );
		//! Set int array type uniform
		void setUniform( 	const char* name,
							const int* val,
							int varDim,
							int count );
		//! Set float type uniform
		void setUniform( 	const char* name,
							const float val );
		//! Set float array type uniform (ex: to set a rotation Matrix varDim=3 count=3)
		void setUniform( 	const char* name,
							const float* val,
							int varDim,
							int count );
		//! Set matrix uniform (to use for matrix 44)
		void setUniformMatrix( 	const char* name,
								const float* mat,
								bool bTranspose );
		//! Set matrix uniform (to use for matrix 33)
		void setUniformMatrix33( 	const char* name,
								const float* mat,
								bool bTranspose );
		//! Set vec2 type uniform
		void setUniformVec2( 	const char* name,
								const Vec2 & a_Value );
		//! Set vec3 type uniform
		void setUniformVec3( 	const char* name,
								const Vec3 & a_Value );
		//! Set vec4 type uniform
		void setUniformVec4( 	const char* name,
								const Vec4 & a_Value );

		//! Returns location of specified attrib (-1 if not found)
		Int getAttribLocation( const char* name ) const;

		//! Bind the specified attrib
		void bindAttribLocation( 	unsigned int index,
									const char* name );

		//! Enable the specified attrib
		Int enableVertexAttribArray( const char* name );
		//! Disable the specified attrib
		void disableVertexAttribArray( Int loc );

		//! Returns the Vertex Shader
		inline const SharedPtr<Shader> & getVertexShader() const
		{	return m_VertexShader;}
		//! Returns the Fragment Shader
		inline const SharedPtr<Shader> & getFragmentShader() const
		{	return m_FragmentShader;}

	private:
		void attachShader( Shader* shader );
		void detachShader( Shader* shader );
		bool link();
		void init( 	const char* a_VertexShaderName,
					const char* a_FragmentShaderName,
					const char* a_GeometryShaderName,
					unsigned int a_GeomInType,
					unsigned int a_GeomOutType,
					bool a_UseIncludeFile );

	protected:
		UInt m_ID;

		SharedPtr< Shader > m_VertexShader;
		SharedPtr< Shader > m_FragmentShader;
		SharedPtr< Shader > m_GeometryShader;

		static const char * m_IncludeCode;
	};
}

#endif // bee_Program_h


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
