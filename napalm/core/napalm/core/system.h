#ifndef _NAPALM_SYSTEM__H_
#define _NAPALM_SYSTEM__H_

#include <string>
#include <assert.h>
#include "util/atomic.hpp"


namespace napalm {


	/*
	 * @class NapalmSystem
	 * @brief System information.
	 */
	class NapalmSystem
	{
	public:

		/*
		 * @brief Singleton access.
		 */
		static NapalmSystem& instance();

		/*
		 * @brief Version info accessors.
		 */
		inline int getMajorVersion() const { return m_majorVersion; }
		inline int getMinorVersion() const { return m_minorVersion; }
		inline int getPatchVersion() const { return m_patchVersion; }
		inline const std::string& getVersionString() const { return m_version; }

		/*
		 * @brief System initialisation.
		 */
		void init();

		/*
		 * @brief
		 * @returns the total number of bytes held in client memory (ie, on cpu) by napalm.
		 */
		inline long getTotalClientBytes() const { return m_totalClientBytes; }

		/*
		 * @brief
		 * Buffers with length less than this threshold will have their contents printed
		 * when displayed as a string.
		 */

		static void count_bytes(long b);

	protected:

		NapalmSystem() { init(); }

	protected:

		std::string 	m_version;
		int 			m_majorVersion;
		int 			m_minorVersion;
		int 			m_patchVersion;
		long 			m_totalClientBytes;
	};


	/*
	 * @brief Force library initialisation
	 */
	static struct _napalm_init
	{
		_napalm_init();
	} _napalm_init_inst;

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
