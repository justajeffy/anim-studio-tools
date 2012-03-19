/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/py/lib/gl/PyProgram.h $"
 * SVN_META_ID = "$Id: PyProgram.h 41922 2010-08-10 03:46:41Z allan.johns $"
 */

#ifndef bee_py_PyProgram_h
#define bee_py_PyProgram_h

#include <string>
#include <boost/shared_ptr.hpp>
#include <kernel/types.h>

//----------------------------------------------------------------------------
namespace bee
{
	//----------------------------------------------------------------------------
	class Program;
	//----------------------------------------------------------------------------
	namespace py
	{
		//----------------------------------------------------------------------------
		//! a texture map
		class Program
		{
		public:
			//! default constructor
			Program();

			//! default destructor
			~Program();

			//! load from a file path
			void read(	const std::string& i_VertexShaderPath,
						const std::string& i_FragmentShaderPath );

			//! load from a file path
			void read(	const std::string& i_VertexShaderPath,
						const std::string& i_FragmentShaderPath,
						const std::string& i_GeometryShaderPath,
						unsigned int i_GeomInType,
						unsigned int i_GeomOutType );

			//! use with a drawing context
			void use();

			//! un-use (when finished drawing)
			void release();

			//! to pass to a texture
			bee::Program* getProgram() const;

			//! some setUniform
			void setUniform( const std::string & name,
							 float x );
			void setUniform( const std::string & name,
							 float x, float y );
			void setUniform( const std::string & name,
							 float x, float y, float z );
			void setUniform( const std::string & name,
							 float x, float y, float z, float w );

		private:

			//! the underlying bee program
			boost::shared_ptr< bee::Program > m_program;
		};

	} // namespace py
} // namespace bee

#endif // bee_py_PyProgram_h


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
