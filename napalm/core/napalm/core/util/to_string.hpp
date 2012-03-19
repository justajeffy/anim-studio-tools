#ifndef _NAPALM_UTIL_TO_STRING__H_
#define _NAPALM_UTIL_TO_STRING__H_

#include <sstream>
#include <vector>
#include <set>
#include "bimath.h"
#include "../typelabels.h"


namespace napalm { namespace util {


	enum StringMode {
		DEFAULT,
		PYTHON,
		TUPLES
	};

	/*
	 * Implements a common string-representation function for all types. Serves to either
	 * disambiguate some types (eg floats have an 'f' appended), or to provide better
	 * strings for external types (such as Imath::Matrix, whos << operator doesn't give
	 * very nice output).
	 */

	template<typename T>
	struct to_string
	{
		static std::string value(const T& t, StringMode a_Mode = DEFAULT)
		{
			std::ostringstream strm;
			strm << t;
			return strm.str();
		}
	};

	// Specialization for <Object> is in object.h

	template<>
	struct to_string<bool>
	{
		static std::string value(const bool& b, StringMode a_Mode = DEFAULT) {
			return (b)? "True" : "False";
		}
	};

	template<>
	struct to_string<float>
	{
		static std::string value(const float& f, StringMode a_Mode = DEFAULT)
		{
			std::ostringstream strm;
			strm << f;
			if(strm.str().rfind('.') == std::string::npos)
				strm << '.';
			if( a_Mode == DEFAULT )
			{
				strm << 'f';
			}
			return strm.str();
		}
	};

	template<>
	struct to_string<half>
	{
		static std::string value(const half& h, StringMode a_Mode = DEFAULT)
		{
			std::ostringstream strm;
			strm << h.operator float();
			if(strm.str().rfind('.') == std::string::npos)
				strm << '.';
			if( a_Mode == DEFAULT )
			{
				strm << 'h';
			}
			return strm.str();
		}
	};

	template<>
	struct to_string<double>
	{
		static std::string value(const double& d, StringMode a_Mode = DEFAULT)
		{
			std::ostringstream strm;
			strm << d;
			if(strm.str().rfind('.') == std::string::npos)
				strm << ".0";
			return strm.str();
		}
	};

	template<>
	struct to_string<std::string>
	{
		static std::string value(const std::string& s, StringMode a_Mode = DEFAULT) {
			return std::string("\"") + s + std::string("\"");
		}
	};

	template<>
	struct to_string<char>
	{
		static std::string value(const char& c, StringMode a_Mode = DEFAULT) {
			std::ostringstream strm;
			strm << static_cast<int>(c);
			if( a_Mode == DEFAULT )
			{
				strm << 'c';
			}
			return strm.str();
		}
	};

	template<>
	struct to_string<unsigned char>
	{
		static std::string value(const unsigned char& c, StringMode a_Mode = DEFAULT) {
			std::ostringstream strm;
			strm << static_cast<unsigned int>(c);
			if( a_Mode == DEFAULT )
			{
				strm << "uc";
			}
			return strm.str();
		}
	};

	template<>
	struct to_string<short>
	{
		static std::string value(const short& s, StringMode a_Mode = DEFAULT) {
			std::ostringstream strm;
			strm << s;
			if( a_Mode == DEFAULT )
			{
				strm << 's';
			}
			return strm.str();
		}
	};

	template<>
	struct to_string<unsigned short>
	{
		static std::string value(const unsigned short& s, StringMode a_Mode = DEFAULT) {
			std::ostringstream strm;
			strm << s;
			if( a_Mode == DEFAULT )
			{
				strm << "us";
			}
			return strm.str();
		}
	};

	template<>
	struct to_string<unsigned int>
	{
		static std::string value(const unsigned int& i, StringMode a_Mode = DEFAULT) {
			std::ostringstream strm;
			strm << i;
			if(a_Mode == DEFAULT)
			{
				strm << 'u';
			}
			return strm.str();
		}
	};

	template<typename T>
	struct to_string<Imath::Vec2<T> >
	{
		static std::string value(const Imath::Vec2<T>& v, StringMode a_Mode = DEFAULT)
		{
			std::ostringstream strm;
			if(a_Mode != TUPLES)
				strm << type_label<Imath::Vec2<T> >::value() << v;

			strm << '(' << v.x << ", " << v.y << ')';
			return strm.str();
		}
	};

	template<typename T>
	struct to_string<Imath::Vec3<T> >
	{
		static std::string value(const Imath::Vec3<T>& v, StringMode a_Mode = DEFAULT)
		{
			std::ostringstream strm;
			if(a_Mode != TUPLES)
				strm << type_label<Imath::Vec3<T> >::value();

			strm << '(' << v.x << ", " << v.y << ", " << v.z << ')';
			return strm.str();
		}
	};

	template<typename T>
	struct to_string<Imath::Vec4<T> >
	{
		static std::string value(const Imath::Vec4<T>& v, StringMode a_Mode = DEFAULT)
		{
			std::ostringstream strm;
			if(a_Mode != TUPLES)
				strm << type_label<Imath::Vec4<T> >::value();

			strm << '(' << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ')';
			return strm.str();
		}
	};

	template<typename T>
	struct to_string<Imath::Box<T> >
	{
		static std::string value(const Imath::Box<T>& b, StringMode a_Mode = DEFAULT)
		{
			std::ostringstream strm;
			if(a_Mode != TUPLES)
				strm << type_label<Imath::Box<T> >::value();

			StringMode vecm = (a_Mode==PYTHON)? PYTHON : TUPLES;
			strm << '(' << to_string<T>::value(b.min, vecm) << ", "
				<< to_string<T>::value(b.max, vecm) << ')';

			return strm.str();
		}
	};

	template<typename T, typename Q>
	struct to_string<std::vector<T,Q> >
	{
		typedef typename std::vector<T,Q>::const_iterator const_iterator;
		static std::string value(const std::vector<T,Q>& v, StringMode a_Mode = DEFAULT)
		{
			std::ostringstream strm;
			strm << "[ ";
			bool first = true;
			const_iterator it = v.begin();
			for( ; it != v.end(); ++it )
			{
				if( !first )
				{
					strm << ", ";
				}
				strm << to_string<T>::value(*it);
				first = false;
			}
			strm << " ]";
			return strm.str();
		}
	};

	template<typename T, typename Q, typename R>
	struct to_string<std::set<T,Q,R> >
	{
		typedef typename std::set<T,Q,R>::const_iterator const_iterator;
		static std::string value(const std::set<T,Q,R>& v, StringMode a_Mode = DEFAULT)
		{
			std::ostringstream strm;
			strm << "set([ ";
			bool first = true;
			const_iterator it = v.begin();
			for( ; it != v.end(); ++it )
			{
				if( !first )
				{
					strm << ", ";
				}
				strm << to_string<T>::value(*it);
				first = false;
			}
			strm << " ])";
			return strm.str();
		}
	};

	// Note: the PYTHON mode for matrices is intended to trigger the sequence contructor;
	// thus the whole lot is wrapped in an extra set of brackets. See pimath for more details.
	template<typename T>
	struct to_string<Imath::Matrix33<T> >
	{
		static std::string value(const Imath::Matrix33<T>& m, StringMode a_Mode = DEFAULT)
		{
			std::ostringstream strm;
			if( a_Mode != TUPLES )
				strm << type_label<Imath::Matrix33<T> >::value();

			strm <<
				( (a_Mode != DEFAULT) ? "(((" : "((" )
				<< m.x[0][0] << ", " << m.x[0][1] << ", " << m.x[0][2] << "), ("
				<< m.x[1][0] << ", " << m.x[1][1] << ", " << m.x[1][2] << "), ("
				<< m.x[2][0] << ", " << m.x[2][1] << ", " << m.x[2][2] <<
				( (a_Mode != DEFAULT) ? ")))" : "))" );
			return strm.str();
		}
	};

	template<typename T>
	struct to_string<Imath::Matrix44<T> >
	{
		static std::string value(const Imath::Matrix44<T>& m, StringMode a_Mode = DEFAULT)
		{
			std::ostringstream strm;
			if( a_Mode != TUPLES )
				strm << type_label<Imath::Matrix44<T> >::value();

			strm <<
				( (a_Mode != DEFAULT) ? "(((" : "((" )
				<< m.x[0][0] << ", " << m.x[0][1] << ", " << m.x[0][2] << ", " << m.x[0][3] << "), ("
				<< m.x[1][0] << ", " << m.x[1][1] << ", " << m.x[1][2] << ", " << m.x[1][3] << "), ("
				<< m.x[2][0] << ", " << m.x[2][1] << ", " << m.x[2][2] << ", " << m.x[2][3] << "), ("
				<< m.x[3][0] << ", " << m.x[3][1] << ", " << m.x[3][2] << ", " << m.x[3][3] <<
				( (a_Mode != DEFAULT) ? ")))" : "))" );
			return strm.str();
		}
	};

} } // ns


#endif
