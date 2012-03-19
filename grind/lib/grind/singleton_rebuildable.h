/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: singleton_rebuildable.h 91641 2011-07-18 03:13:38Z chris.bone $"
 */

#ifndef rebuildable_singleton_h
#define rebuildable_singleton_h

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/scoped_ptr.hpp>

namespace drd {

//! a thread safe singleton that can be created and destroyed as required
template<class T>
class RebuildableSingleton
: private boost::noncopyable
{
public:

	//! allow access to the single instance
	static T& instance() {
		boost::mutex::scoped_lock l( m_Mtx );
		if( t == NULL ) init();
		return *t;
	}

	//! destroy the single instance
	static void destroy() {
		boost::mutex::scoped_lock l( m_Mtx );
		kill();
	}

protected:
	~RebuildableSingleton() {}
	RebuildableSingleton() {}

private:

	//! set up the one instance
	static void init() // never throws
	{
		t.reset( new T() );
	}

	//! destroy the one instance
	static void kill() // never throws
	{
		t.reset( 0 );
	}

	static boost::scoped_ptr<T> t;
	static boost::mutex m_Mtx;
};

} // drd namespace


template<class T> boost::scoped_ptr<T> drd::RebuildableSingleton<T>::t(NULL);
template<class T> boost::mutex drd::RebuildableSingleton<T>::m_Mtx;

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
