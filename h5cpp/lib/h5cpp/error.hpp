#ifndef _H5CPP_ERROR__H_
#define _H5CPP_ERROR__H_

#include <sstream>
#include <assert.h>
#include <stdexcept>
#include <hdf5/H5Epublic.h>


/*
 * exception-throwing macros that can take an ostream-like format message
 */
#define H5CPP_THROW_HDF5( MSG )						\
	{												\
		std::stringstream msg;						\
		msg << MSG;									\
		throw ::h5cpp::hdf5_error(msg.str());		\
	}

#define H5CPP_THROW( MSG )							\
	{												\
		std::stringstream msg;						\
		msg << MSG;									\
		throw ::h5cpp::h5cpp_error(msg.str());		\
	}

/*
 * throw an exception when hdf5 errors occur
 */
#define H5CPP_ERR_ON_NEG(expr) 						\
	::h5cpp::detail::error_on_neg(expr, #expr, __func__, __FILE__, __LINE__)


namespace h5cpp
{

	/*
	 * @class h5cpp_error
	 * @brief exception for errors that have come from h5cpp itself
	 */
	class h5cpp_error : public std::runtime_error
	{
	public:
		explicit h5cpp_error(const std::string& s): std::runtime_error(s){}
	};


	/*
	 * @class hdf5_error
	 * @brief exception for errors that have come from hdf5
	 */
	class hdf5_error : public h5cpp_error
	{
	public:
		hdf5_error(const std::string& s):h5cpp_error(s){}
	};


	namespace detail
	{

		void throw_hdf5_error(const std::string& msg);

		template<typename T>
		T error_on_neg(T val, const char* statement, const char* function, const char* file, int line)
		{
			if(val < 0)
			{
				std::stringstream msg;
				msg << statement <<  " failed in " << function << ", file " << file << " line " << line;

				throw_hdf5_error(msg.str());
			}
			return val;
		}

		/*
		 * @class scoped_quiet_hdf5_errors
		 * @brief suppresses hdf5 errors from writing to stderr - we want to catch them
		 * as exceptions instead.
		 */
		class scoped_suppress_hdf5_errors
		{
		public:
			scoped_suppress_hdf5_errors();
			~scoped_suppress_hdf5_errors();
		private:
			H5E_auto_t m_prevFunc;
			void* m_prevClientData;
		};
	}

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
