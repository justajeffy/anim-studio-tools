/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/kernel/log.cpp $"
 * SVN_META_ID = "$Id: log.cpp 17302 2009-11-18 06:20:42Z david.morris $"
 */

#define MaxStringLength 8192
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "kernel/log.h"

#ifdef WIN32
#define snprintf _snprintf
#endif

using namespace std;

//-----------------------------------------------------------------------------
namespace
{
    static int s_LogLine;
    static const char * s_LogFile;
    static const char * s_LogFunc;
    static int s_LogLevel = DEFAULT_LOG_LEVEL;


//-----------------------------------------------------------------------------
    void __Log( const char * a_Str )
    {
        cout << a_Str << endl;
#ifdef WIN32
		OutputDebugStringA( a_Str );
		OutputDebugStringA( "\n" );
#endif
    }

//-----------------------------------------------------------------------------
    void
    _LogInt( const char * a_Format, va_list a_VA )
    {
        char buf [ MaxStringLength ];
        vsnprintf( buf, MaxStringLength, a_Format, a_VA );
        __Log( buf );
    }


//-----------------------------------------------------------------------------
    void _LogInt( const char * a_Format, ... )
    {
        va_list va;
        va_start( va, a_Format );
        _LogInt( a_Format, va );
        va_end( va );
    }

//-----------------------------------------------------------------------------
    void _Log( const char * a_Category, const char * a_Format, va_list a_VA )
    {
        char buf0 [ MaxStringLength ];
        char buf1 [ MaxStringLength ];
        vsnprintf( buf0, MaxStringLength, a_Format, a_VA );
        snprintf( buf1, MaxStringLength, "%s: [%20s:%-4i:%-12s()] %s", a_Category, s_LogFile, s_LogLine, s_LogFunc, buf0 );
        __Log( buf1 );
    }
} // end of anon namespace

//-----------------------------------------------------------------------------
int LOG_LEVEL()
{
	return s_LogLevel;
}

//-----------------------------------------------------------------------------
void LOG_LEVEL( int a_LogLevel )
{
	s_LogLevel = a_LogLevel;
}

//-----------------------------------------------------------------------------
bool
AssertFailure( const char * a_Txt,
                            const char * a_Msg,
                            const char * a_File,
                            unsigned int a_Line)
{
    char buf [ MaxStringLength ];
    snprintf( buf, MaxStringLength, "%s\n%s\nFAILED IN FILE [%s:%d]\n", a_Msg, a_Txt, a_File, a_Line );
    __Log( buf );
    return true;
}

//-----------------------------------------------------------------------------
const char *
AssertFailureFormat( const char * a_Format, ... )
{
    static char str [ MaxStringLength ];
    va_list va;
    va_start( va, a_Format );
    vsnprintf( str, MaxStringLength, a_Format, va );
    va_end( va );
    return str;
}

//-----------------------------------------------------------------------------
void
_SetLogData( unsigned int a_Line, const char * a_File, const char * a_Function )
{
    s_LogLine = a_Line;
    s_LogFile = a_File;
    s_LogFunc = a_Function;
}

//-----------------------------------------------------------------------------
#define IMPLEMENT_CATEGORY(CAT)                                               \
    void CAT( const char * a_Format, ... )                                    \
    {                                                                         \
        va_list va;                                                           \
        va_start( va, a_Format );                                             \
        _Log( #CAT, a_Format, va );                                           \
        va_end( va );                                                         \
    }

//-----------------------------------------------------------------------------
IMPLEMENT_CATEGORY(DEBG);
IMPLEMENT_CATEGORY(INFO);
IMPLEMENT_CATEGORY(WARN);
IMPLEMENT_CATEGORY(ERRR);
IMPLEMENT_CATEGORY(PRNT);


//------------------------------------------------------------------------------
string
Format( const char * a_Fmt, ... )
{
	char str [ MaxStringLength ];
	va_list va;
	va_start( va, a_Fmt );
	vsnprintf( str, MaxStringLength, a_Fmt, va );
	va_end( va );
	return string( str );
}



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
