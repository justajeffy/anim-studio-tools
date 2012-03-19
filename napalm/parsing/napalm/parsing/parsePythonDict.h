#ifndef _NAPALM_CORE_PARSING_PARSEPYTHONDICT__H_
#define _NAPALM_CORE_PARSING_PARSEPYTHONDICT__H_

#include "napalm/core/Table.h"
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

namespace napalm {

	extern boost::mutex  		g_parse_mtx;
	/*
	 * @brief parsePythonDict
	 * Given a string representation of a python dict, return the equivalent Napalm table.
	 * Supports ints and strings as dict keys, and ints, strings, floats, booleans, lists
	 * and dicts as dict values (lists are converted to zero-indexed dicts). Strings are
	 * *single*-quoted only. Note that boolean values are converted into int: True=1,
	 * False=0. Napalm deliberately has no boolean type.
	 *
	 * @param dict_str The string to parse (eg: "{1:2, 'fee':'fo', 5:[6,7,8]}")
	 * @param perror If non-null, any parsing error info will be written to this string, and
	 * the function will not throw. If null, the function will throw on parser error.
	 * @returns NULL on parser fail, otherwise the resulting Napalm table.
	 */
	 object_table_ptr parsePythonDict(const std::string& dict_str, std::string* perror = NULL);

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
