#ifndef _NAPALM_TYPEDBUFFERBIND__H_
#define _NAPALM_TYPEDBUFFERBIND__H_

#include <boost/python.hpp>
#include <vector>
#include <stdexcept>
#include <sstream>
#include "TypedBuffer.h"
#include "util/type_convert.hpp"
#include "util/safe_convert.hpp"
#include "util/bimath.h"


namespace napalm
{

	namespace bp = boost::python;

	std::size_t verifiedIndex(std::size_t size, int index);


	// specializations for T where T != std::vector<U>
	template<typename T>
	struct TypedBufferBind_VectorizedDependant
	{
		typedef util::type_converter<T>											py_conv_type;
		typedef typename py_conv_type::type										py_elem_type;
		typedef TypedBuffer<T> 													buffer_type;
		typedef boost::shared_ptr<buffer_type> 									buffer_type_ptr;
		typedef bp::class_<buffer_type, bp::bases<Buffer>, buffer_type_ptr>		bp_class_;

		static typename util::type_converter<T>::type
		getItem(const buffer_type& self, int index)
		{
			return self.r()[verifiedIndex(self.size(), index)];
		}

		static void setItem(buffer_type& self, int index,
			const typename util::type_converter<T>::type& val)
		{
			self.rw()[verifiedIndex(self.size(), index)] =
				util::type_converter<T>::from_python(val);
		}

		static bp::list getContents(const buffer_type& self)
		{
			bp::list l;
			typename buffer_type::r_type fr = self.r();
			for(typename buffer_type::r_type::iterator it=fr.begin(); it!=fr.end(); ++it)
				l.append(py_conv_type::to_python(*it));
			return l;
		}

		static void setContents(buffer_type& self, const bp::list& l)
		{
			unsigned int sz = bp::len(l);

			// copy contents into a tmp vector first, in case a list elem extract fails
			std::vector<T> tmpv(sz);
			for(unsigned int i=0; i<sz; ++i)
				tmpv[i] = py_conv_type::from_python(bp::extract<py_elem_type>(l[i]));

			self.resize(sz, true);
			typename buffer_type::w_type fr = self.w();
			std::copy(tmpv.begin(), tmpv.end(), fr.begin());
		}

		static buffer_type* filledInit(unsigned int nelems, const py_elem_type& val)
		{
			return new buffer_type(nelems, py_conv_type::from_python(val));
		}

		static void bind(bp_class_& cl)
		{
			cl.def(bp::init<bp::optional<unsigned int> >());
			cl.def("__init__", bp::make_constructor(filledInit));
			cl.def("__getitem__", getItem);
			cl.def("__setitem__", setItem);
			cl.add_property("contents", getContents, setContents);
		}
	};


	// specializations for T where T == std::vector<U>
	template<typename T, typename Allocator>
	struct TypedBufferBind_VectorizedDependant<std::vector<T,Allocator> >
	{
		typedef util::type_converter<T>											py_conv_type;
		typedef typename py_conv_type::type										py_elemitem_type;
		typedef std::vector<T,Allocator>										T_;
		typedef TypedBuffer<T_> 												buffer_type;
		typedef boost::shared_ptr<buffer_type> 									buffer_type_ptr;
		typedef bp::class_<buffer_type, bp::bases<Buffer>, buffer_type_ptr>		bp_class_;

		static bp::list getItem(const buffer_type& self, int index)
		{
			bp::list l;
			const T_& elem = self.r()[verifiedIndex(self.size(), index)];
			for(typename T_::const_iterator it=elem.begin(); it!=elem.end(); ++it)
				l.append(py_conv_type::to_python(*it));
			return l;
		}

		static void setItem(buffer_type& self, int index, const bp::list& val)
		{
			std::size_t j = verifiedIndex(self.size(), index);

			T_& val_ = self.rw()[j];
			val_.resize(bp::len(val));

			for(long i=0; i<bp::len(val); ++i)
				val_[i] = bp::extract<T>(val[i]);
		}

		static bp::list getContents(const buffer_type& self)
		{
			bp::list l;
			typename buffer_type::r_type fr = self.r();
			for(typename buffer_type::r_type::iterator it=fr.begin(); it!=fr.end(); ++it)
			{
				bp::list l2;
				for(typename T_::const_iterator it2=it->begin(); it2!=it->end(); ++it2)
					l2.append(py_conv_type::to_python(*it2));
				l.append(l2);
			}

			return l;
		}

		static void setContents(buffer_type& self, const bp::list& l)
		{
			// copy contents into a tmp vector first, in case a list elem extract fails
			unsigned int sz = bp::len(l);
			std::vector<T_> tmpv(sz);

			for(unsigned int i=0; i<sz; ++i)
			{
				T_& elem = tmpv[i];
				bp::object pyelem = l[i];

				unsigned int elemsize = bp::len(pyelem);
				elem.resize(elemsize);
				for(unsigned int j=0; j<elemsize; ++j)
					elem[j] = py_conv_type::from_python(bp::extract<py_elemitem_type>(pyelem[j]));
			}

			self.resize(sz, true);
			typename buffer_type::w_type fr = self.w();
			std::copy(tmpv.begin(), tmpv.end(), fr.begin());
		}

		static buffer_type* listInit(unsigned int nelems, const bp::list& l)
		{
			unsigned int elemlen = bp::len(l);
			T_ elem(elemlen);
			for(unsigned int i=0; i<elemlen; ++i)
				elem[i] = py_conv_type::from_python(bp::extract<py_elemitem_type>(l[i]));

			return new buffer_type(nelems, elem);
		}

		static void bind(bp_class_& cl)
		{
			cl.def(bp::init<bp::optional<unsigned int> >());
			cl.def("__init__", bp::make_constructor(listInit));
			cl.def("__getitem__", getItem);
			cl.def("__setitem__", setItem);
			cl.add_property("contents", getContents, setContents);
		}
	};


	template<typename T, typename S, typename Enable=void>
	struct TypedBufferBind_CopyConstr
	{
		typedef TypedBuffer<T> 													buffer_type;
		typedef boost::shared_ptr<buffer_type> 									buffer_type_ptr;
		typedef bp::class_<buffer_type, bp::bases<Buffer>, buffer_type_ptr>		bp_class_;
		static void bind(bp_class_& cl){}
	};


	template<typename T, typename S>
	struct TypedBufferBind_CopyConstr<T, S, typename boost::enable_if<
		util::is_safe_convertible<T,S> >::type>
	{
		typedef TypedBuffer<T> 													buffer_type;
		typedef TypedBuffer<S> 													buffer2_type;
		typedef boost::shared_ptr<buffer_type> 									buffer_type_ptr;
		typedef boost::shared_ptr<buffer2_type> 								buffer2_type_ptr;
		typedef boost::shared_ptr<const buffer2_type> 							c_buffer2_type_ptr;
		typedef bp::class_<buffer_type, bp::bases<Buffer>, buffer_type_ptr>		bp_class_;

		static buffer_type* copyInit(buffer2_type_ptr rhs) {
			return new buffer_type(rhs);
		}

		static void bind(bp_class_& cl) {
			cl.def("__init__", bp::make_constructor(copyInit));
		}
	};


	template<typename T>
	struct TypedBufferBind
	{
		typedef TypedBuffer<T> 													buffer_type;
		typedef boost::shared_ptr<buffer_type> 									buffer_type_ptr;
		typedef bp::class_<buffer_type, bp::bases<Buffer>, buffer_type_ptr>		bp_class_;

		TypedBufferBind(const std::string& name)
		{
			bp_class_ cl(name.c_str());

			#define _NAPALM_TYPE_OP(S, Label) TypedBufferBind_CopyConstr<T,S>::bind(cl);
			#include "types/all.inc"
			#undef _NAPALM_TYPE_OP

			TypedBufferBind_VectorizedDependant<T>::bind(cl);
		}
	};

}

#endif










