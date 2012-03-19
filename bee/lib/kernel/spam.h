/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/kernel/spam.h $"
 * SVN_META_ID = "$Id: spam.h 17302 2009-11-18 06:20:42Z david.morris $"
 */

#ifndef bee_spam_h
#define bee_spam_h
#pragma once

#include <iostream>

namespace bee
{
	#define SPAMt(t,s) std::cout << t << #s << " = " << s << std::endl
	#define SPAM(s) std::cout << #s << " = " << s << std::endl
}

#endif // bee_spam_h


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
