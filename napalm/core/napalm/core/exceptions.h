#ifndef _NAPALM_EXCEPTIONS__H_
#define _NAPALM_EXCEPTIONS__H_

#include <exception>
#include <stdexcept>
#include <sstream>


namespace napalm {

	/*
	 * Helper macro for raising exceptions
	 */
	#define NAPALM_THROW(Exception, Msg) { 	\
		std::stringstream strm;				\
		strm << Msg;						\
		throw Exception(strm.str());		\
	}


	/*
	 * All napalm exceptions derive from this
	 */
	class NapalmError : public std::runtime_error
	{
	public:
		NapalmError(const std::string& msg)
		: std::runtime_error(msg){}
	};


	/*
	 * File access error
	 */
	class NapalmFileError : public NapalmError
	{
	public:
		NapalmFileError(const std::string& msg)
		: NapalmError(msg){}
	};


	/*
	 * There was an error reading or writing data from an archive
	 */
	class NapalmSerializeError : public NapalmError
	{
	public:
		NapalmSerializeError(const std::string& msg)
		: NapalmError(msg){}
	};


}


#endif


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
