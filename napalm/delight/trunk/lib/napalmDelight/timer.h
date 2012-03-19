#ifndef TIMER_H_
#define TIMER_H_

#include <sys/time.h>
#include <iostream>

namespace wool_lint {

//-------------------------------------------------------------------------------------------------
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
