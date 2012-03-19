/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/kernel/log.h $"
 * SVN_META_ID = "$Id: log.h 17302 2009-11-18 06:20:42Z david.morris $"
 */


#ifndef bee_log_h
#define bee_log_h
#pragma once

// change this to reflect what logging level you want
#define DEFAULT_LOG_LEVEL 1

int LOG_LEVEL();
void LOG_LEVEL(int l);

bool AssertFailure( const char * a_Txt, const char * a_Msg, const char * a_File, unsigned int a_Line );
const char * AssertFailureFormat( const char * a_Format, ... );

#define ASSERTM(exp,msg) if (!(exp)) { if ( AssertFailure( #exp, AssertFailureFormat msg, __FILE__, __LINE__ ) ) {} } else {}

#define DEBG_LEVEL 1
#define INFO_LEVEL 2
#define WARN_LEVEL 3
#define ERRR_LEVEL 4
// Always print
#define PRNT_LEVEL 100

inline void HandleGlError( int a_Error )
{
	a_Error;
#ifdef WIN32
	__asm { int 3 }
#endif
}

//-----------------------------------------------------------------------------
void DEBG( const char * a_Format, ... );
void INFO( const char * a_Format, ... );
void WARN( const char * a_Format, ... );
void ERRR( const char * a_Format, ... );
void PRNT( const char * a_Format, ... );
void _SetLogData( unsigned int a_Line, const char * a_File, const char * a_Function );
#define LOG(CAT,X) if ( CAT##_LEVEL >= LOG_LEVEL() ) { _SetLogData( __LINE__, __FILE__, __FUNCTION__ ); CAT X ; } else { /*fprintf( stderr, "Wrong log level: %d>=%d\n", CAT##_LEVEL, LOG_LEVEL() );*/ }
#define LOG_IF_GL_ERROR(CAT,X)                                                              \
if ( CAT##_LEVEL >= LOG_LEVEL() ) {                                                         \
		int error = glGetError();                                                           \
		if ( error != GL_NO_ERROR ) {                                                       \
			HandleGlError( error );															\
			LOG( CAT, X );                                                                  \
			LOG( CAT, ( "openGL ERROR, possible cause: [%i] [%s]",                          \
				error, gluErrorString( error ) ) );                                         \
		}                                                                                   \
} else {}

//-----------------------------------------------------------------------------
std::string Format( const char * a_Fmt, ... );

#endif // bee_log_h


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
