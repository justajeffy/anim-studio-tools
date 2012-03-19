/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: timer.h 45312 2010-09-09 06:11:02Z chris.cooper $"
 */

#ifndef grind_timer_h
#define grind_timer_h

#include <sys/time.h>
#include <iostream>

namespace grind
{
	//-------------------------------------------------------------------------------------------------
	//! cpu timer adapted from OpenCurrent
	struct CPUTimer
	{
	  double _start_time, _end_time;

	  CPUTimer() { }
	  ~CPUTimer() { }

	  void start() {
	    struct timeval t;
	    gettimeofday(&t, 0);
	    _start_time = t.tv_sec + (1e-6 * t.tv_usec);
	  }
	  void stop()  {
	    struct timeval t;
	    gettimeofday(&t, 0);
	    _end_time = t.tv_sec + (1e-6 * t.tv_usec);
	  }

	  float elapsed_sec() {
	    return (float) (_end_time - _start_time);
	  }

	  float elapsed_ms()
	  {
	    return (float) 1000 * (_end_time - _start_time);
	  }

	  const char* timer_type()
	  {
		  return "cpu";
	  }
	};

	//-------------------------------------------------------------------------------------------------
	//! gpu timer adapted from OpenCurrent
	struct GPUTimer
	{
	  void *e_start;
	  void *e_stop;

	  GPUTimer();
	  ~GPUTimer();

	  void start();
	  void stop();

	  float elapsed_ms();
	  float elapsed_sec() { return elapsed_ms() / 1000.0f; }

	  const char* timer_type()
	  {
		  return "gpu";
	  }
	};

	//-------------------------------------------------------------------------------------------------
	template< typename TIMER >
	struct ScopeTimer
	{
		ScopeTimer( const char* decoration )
		: m_Decoration( decoration )
		{
			m_Timer.start();
		}

		~ScopeTimer()
		{
			m_Timer.stop();
			std::cerr << "TIMER:  " << m_Decoration << " (" << m_Timer.timer_type() << "): " << m_Timer.elapsed_ms() << "ms" << std::endl;
		}

		const char* m_Decoration;
		TIMER m_Timer;
	};

}// grind


//-------------------------------------------------------------------------------------------------
#define CPU_TIMER(decoration,fn) \
{ \
	CPUTimer timer; \
	timer.start(); \
	fn; \
	timer.stop(); \
	DRD_LOG_INFO( L, decoration << " (cpu): " << timer.elapsed_ms() << "ms" ); \
}

//-------------------------------------------------------------------------------------------------
#define GPU_TIMER(decoration,fn) \
{ \
	GPUTimer timer; \
	timer.start(); \
	fn; \
	timer.stop(); \
	DRD_LOG_INFO( L, decoration << " (gpu): " << timer.elapsed_ms() << "ms" ); \
}

#ifdef __DEVICE_EMULATION__
#define GRIND_TIMER(decoration,fn) CPU_TIMER(decoration,fn)
#else
#define GRIND_TIMER(decoration,fn) GPU_TIMER(decoration,fn)
#endif

//-------------------------------------------------------------------------------------------------
#define CPU_SCOPE_TIMER(decoration) grind::ScopeTimer<grind::CPUTimer> _timer(decoration)
#define GPU_SCOPE_TIMER(decoration) grind::ScopeTimer<grind::GPUTimer> _timer(decoration)

#endif /* grind_timer_h */


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
