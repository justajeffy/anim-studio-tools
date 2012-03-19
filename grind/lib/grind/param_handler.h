/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: param_handler.h 45312 2010-09-09 06:11:02Z chris.cooper $"
 */

#ifndef grind_param_handler_h
#define grind_param_handler_h

#include <boost/program_options.hpp>
//-------------------------------------------------------------------------------------------------

class ParamHandler
{
public:
	//! access the description
	boost::program_options::options_description& getDesc() { return m_Desc; }

	//! set the value for a named param
	void setParam( const std::string& param, const std::string& val );

	//! list the available params
	void listParams()
	{
		DRD_LOG_INFO( L, m_Desc );
	}

private:
	//! the program options description
	boost::program_options::options_description m_Desc;

	//! variable map
	boost::program_options::variables_map m_Vm;
};

#endif /* grind_param_handler_h */


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
