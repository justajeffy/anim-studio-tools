#include <napalm/core/core.h>
#include <boost/assign/list_of.hpp>
#include <algorithm>


using namespace napalm;


#ifdef NDEBUG
#define __ASSERT(cond) 						\
	if(!(cond)) { 							\
		std::cout << "failed on line " << __LINE__ << std::endl; \
		exit(1); }
#else
#define __ASSERT(cond) assert(cond);
#endif


void printEntry( const std::pair< TableKey, object_ptr > & pair )
{
	std::cout << *pair.first << ":" << *pair.second << std::endl;
}


struct less
{
	bool operator()( attrib_ptr a, attrib_ptr b ) {
		return *a < *b;
	}
};

template<typename TO>
void runExtractTest(attrib_table_ptr a_From, const char * type_name )
{
	for( AttribTable::typed_iterator<std::string> it = a_From->typed_begin<std::string>();
			it != a_From->end(); ++it )
	{
		TO t;
		bool val = it.extractValue( t );
		if( val )
		{
			if( it.key() != type_name)
			{
				std::cout << "extracted "<<it.key()<<" as "<<type_name<<" ("<<util::to_string<TO>::value(t)<<")"<<std::endl;
			}
		}
		else
		{
			if( it.key() == type_name )
			{
				std::cout << "could not extract "<<it.key()<<" as "<<type_name<<std::endl;
				// Always fail here, types should always be extractable as themselves.
				__ASSERT(false);
			}
		}
	}
}


int main(int arc, char** argv)
{
	// temp
	// ##########################################

	IntBufferPtr b = IntBufferPtr(new IntBuffer(10, 55));
	V3iBufferPtr b2 = V3iBufferPtr(new V3iBuffer(b));

	for(IntBuffer::iterator it=b->rw_begin(); it!=b->rw_end(); ++it)
		__ASSERT( *it == 55 );

	// ##########################################


	// create tables
	object_table_ptr t(new ObjectTable());
	attrib_table_ptr a(new AttribTable());


	// test simple setEntry variations
	if(1)
	{
		// write to attrib table
		a->setEntry("typish", "buffery");

		AttribTable::iterator it = a->begin();
		__ASSERT( *it.getValue<std::string>() == std::string("buffery") );
		__ASSERT( it.getValue<int>() == NULL );
		std::string temp;
		__ASSERT( it.extractValue(temp) == true );
		__ASSERT( temp == std::string("buffery") );
		int itemp;
		__ASSERT( it.extractValue(itemp) == false );
		__ASSERT( *it->first == *Wrapper<Attribute>(std::string("typish") ) );
		Attribute *attrib ( it->second.get() );
		TypedAttribute<std::string> *ta ( static_cast<TypedAttribute<std::string>*>(attrib) );
		__ASSERT( ta != NULL );
		__ASSERT( ta->value() == std::string("buffery"));
		__ASSERT( it == a->begin() );
		__ASSERT( it++ == a->begin() );
		__ASSERT( it == a->end() );
		it = a->begin();
		it.setValue(17);
		__ASSERT( it.getValue<std::string>() == NULL );
		__ASSERT( *it.getValue<int>() == 17 );
		it.setValue(std::string("buffery") );

		a->setEntry(5, 500);

		{
			boost::variant<float,double> v(4.5f);
			a->setEntry("f", v);
		}

		{
			boost::shared_ptr<IntAttrib> i(new IntAttrib(56));
			a->setEntry("fifty-six", i);
		}

		object_rawptr_set printed;
		for( it = a->begin(); it != a->end(); ++it )
		{
			it->first->str(std::cout, printed);
			std::cout << ":";
			it->second->str(std::cout, printed);
			std::cout << std::endl;
		}

		// write to object table
		V3fBufferPtr buf(new V3fBuffer(100));
		buf->setAttribs(a);

		t->setEntry("P", buf);
		t->setEntry("A", a);
		t->setEntry("str", std::string("hello!"));

		{
			boost::variant<int,float,V3fBufferPtr> v(6.66f);
			t->setEntry(6, v);
		}

		t->setEntry("t2", make_clone(t));

		for( Table<Object>::typed_iterator<std::string> it = t->typed_begin<std::string>();
				it != t->end(); ++it )
		{
			std::cout << "Typed iterator: " << it.key() << ":" << *it->second << std::endl;
		}
	}


	// test hasEntry variations
	if(1)
	{
		__ASSERT(t->hasEntry("P"));
		__ASSERT(t->hasEntry(6));
		__ASSERT(!t->hasEntry("foo"));

		// Return by reference test
		__ASSERT(t->getEntry<std::string>("str"));
		__ASSERT(*t->getEntry<std::string>("str") == std::string("hello!"));
		__ASSERT(t->getEntry<V3fBuffer>("P"));
		__ASSERT(t->getEntry<V3fBuffer>("P")->size() == 100);
		__ASSERT(*(t->getEntry<V3fBuffer>("P")->getAttribs()->getEntry<int>(5)) == 500)
		__ASSERT(t->getEntry<V3fBuffer>("P")->getAttribs()->getEntry<int>(17) == NULL)

		// Extract entry tests
		std::string str;
		float f;
		int i;
		__ASSERT(t->extractEntry("str", f) == false);
		__ASSERT(t->extractEntry("str", str) == true);
		__ASSERT(a->extractEntry(5, str) == false);
		__ASSERT(a->extractEntry(5, i) == true);
		__ASSERT(i == 500);
		__ASSERT(a->extractEntry(5, f) == true);
		__ASSERT(f == 500);
		__ASSERT(a->extractEntry<int>(5) == 500);
		__ASSERT(a->extractEntry<float>(5) == 500.f);
		__ASSERT(t->extractEntry<std::string>("str") == std::string("hello!"));
		__ASSERT(t->extractEntryWithDefault("str", 17) == 17);
		__ASSERT(t->extractEntry("nonexistant", f) == false);

		__ASSERT(t->hasEntryPath(boost::assign::list_of<TableKey>("P")));
		__ASSERT(t->hasEntryPath(boost::assign::list_of<TableKey>("A")("typish")));
		__ASSERT(t->hasEntryPath(boost::assign::list_of<TableKey>("P")(5)));
		__ASSERT(t->hasEntryPath(boost::assign::list_of<TableKey>("A")("f")));
		__ASSERT(t->hasEntryPath(boost::assign::list_of<TableKey>("t2")));
		__ASSERT(t->hasEntryPath(boost::assign::list_of<TableKey>("t2")(6)));
		__ASSERT(t->hasEntryPath(boost::assign::list_of("t2")("P")("typish")));
		__ASSERT(t->hasEntryPath(boost::assign::list_of("t2")("A")("fifty-six")));
		__ASSERT(!t->hasEntryPath(boost::assign::list_of("foo")));
		__ASSERT(!t->hasEntryPath(boost::assign::list_of("A")("foo")));

		{
			table_key_vector keypath = boost::assign::list_of<TableKey>("t2")(6);
			__ASSERT(t->hasEntryPath(keypath));
		}
	}


	// test getEntry variations
	if(1)
	{
		int i;
		float f;
		std::string s;
		V3fBufferPtr b;
		V3fBufferCPtr cb;
		object_table_ptr t2;
		c_object_table_ptr ct2;
		attrib_table_ptr a2;
		c_attrib_table_ptr ca2;

		__ASSERT(t->getEntry(6, f));
		__ASSERT(t->getEntry("str", s));
		__ASSERT(t->getEntry("P", b));
		__ASSERT(t->getEntry("P", cb));
		__ASSERT(t->getEntry("A", a2));
		__ASSERT(t->getEntry("A", ca2));
		__ASSERT(t->getEntry("t2", t2));
		__ASSERT(t->getEntry("t2", ct2));
		__ASSERT(!t->getEntry("foo", f));
		__ASSERT(!t->getEntry(6, i));

		__ASSERT(t->getEntryPath(boost::assign::list_of<TableKey>("P"), b));
		__ASSERT(t->getEntryPath(boost::assign::list_of<TableKey>("P")("typish"), s));
		__ASSERT(t->getEntryPath(boost::assign::list_of<TableKey>("A")(5), i));
		__ASSERT(t->getEntryPath(boost::assign::list_of<TableKey>("P")("f"), f));
		__ASSERT(t->getEntryPath(boost::assign::list_of<TableKey>("t2"), ct2));
		__ASSERT(t->getEntryPath(boost::assign::list_of<TableKey>("t2")(6), f));
		__ASSERT(t->getEntryPath(boost::assign::list_of("t2")("P")("typish"), s));
		__ASSERT(!t->getEntryPath(boost::assign::list_of("foo"), i));
		__ASSERT(!t->getEntryPath(boost::assign::list_of("P")("foo"), cb));
		__ASSERT(!t->getEntryPath(boost::assign::list_of("A")("typish"), t2));
		__ASSERT(!t->getEntryPath(boost::assign::list_of("P")("typish")("blah"), s));

		{
			boost::variant<int,float> v;
			__ASSERT(t->getEntry(6, v));
			__ASSERT(v.which() == 1);

			__ASSERT(t->getEntryPath(boost::assign::list_of<TableKey>("t2")(6), v));
			__ASSERT(v.which() == 1);
		}

		{
			boost::variant<float,std::string,V3fBufferPtr,char> v;
			__ASSERT(t->getEntry("P", v));
			__ASSERT(v.which() == 2);

			__ASSERT(t->getEntryPath(boost::assign::list_of<TableKey>("P"), v));
			__ASSERT(v.which() == 2);
		}

		{
			boost::variant<c_object_table_ptr,double> v;
			__ASSERT(t->getEntry("t2", v));
			__ASSERT(v.which() == 0);

			__ASSERT(t->getEntryPath(boost::assign::list_of<TableKey>("t2"), v));
			__ASSERT(v.which() == 0);
		}

		{
			boost::variant<int,float,std::string> v;
			__ASSERT(!t->getEntry("P", v));
			__ASSERT(!t->getEntryPath(boost::assign::list_of<TableKey>("P"), v));
		}

		{
			boost::variant<int,float> v;
			__ASSERT(!a->getEntry("typish", v));
			__ASSERT(a->getEntry(5, v));
			__ASSERT(!a->getEntry("foo", v));
		}
	}


	// test recursive setEntry
	if(1)
	{
		{
			std::string s;
			__ASSERT(t->setEntryPath(boost::assign::list_of("do")("dee")("doo")("da")("day"), std::string("blah"), true));
			__ASSERT(t->getEntryPath(boost::assign::list_of("do")("dee")("doo")("da")("day"), s));
			__ASSERT(t->getEntryPathT(s, "do", "dee", "doo", "da", "day"));
		}

		{
			double d;
			__ASSERT(t->setEntryPath(boost::assign::list_of<TableKey>("str"), 55.0, false));
			__ASSERT(t->getEntryPath(boost::assign::list_of<TableKey>("str"), d));
		}

		{
			char c;
			__ASSERT(t->setEntryPath(boost::assign::list_of("A")("f"), 'Q', false));
			__ASSERT(t->getEntryPath(boost::assign::list_of("A")("f"), c));
		}

		{
			char c;
			__ASSERT(t->setEntryPath(boost::assign::list_of("P")("f"), 'Q', false));
			__ASSERT(t->getEntryPath(boost::assign::list_of("P")("f"), c));
		}

		{
			int i;
			__ASSERT(!t->setEntryPath(boost::assign::list_of("foo")("bah"), 44, false));
			__ASSERT(t->setEntryPath(boost::assign::list_of("foo")("bah"), 44, true));
			__ASSERT(t->getEntryPath(boost::assign::list_of("foo")("bah"), i));
			__ASSERT(t->getEntryPathT(i, "foo", "bah"));
		}

		{
			IntBufferPtr b(new IntBuffer(10));
			__ASSERT(!t->setEntryPath(boost::assign::list_of<TableKey>("A")("f"), b, false));
			__ASSERT(t->setEntryPath(boost::assign::list_of("A")("f"), b, true)); // 1*
			__ASSERT(t->getEntryPath(boost::assign::list_of("A")("f"), b));

			c_object_table_ptr tbl;
			__ASSERT(t->getEntry("A", tbl)); // changed to object-table due to 1*

			// same deal but with buffer
			__ASSERT(t->setEntryPath(boost::assign::list_of("P")("f"), b, true));
			__ASSERT(t->getEntryPath(boost::assign::list_of("P")("f"), b));
			__ASSERT(t->getEntry("P", tbl));
			__ASSERT((*t)["P"] == tbl);
		}

		{
			boost::variant<int,std::string> v("hello");
			__ASSERT(t->setEntryPath(boost::assign::list_of<TableKey>("P")(5), v, false));

			std::string s;
			__ASSERT(t->getEntryPath(boost::assign::list_of<TableKey>("P")(5), s));

			boost::variant<int,std::string> v2;
			__ASSERT(t->getEntryPath(boost::assign::list_of<TableKey>("P")(5), v2));
			__ASSERT(v == v2);
		}

		// Unusually-typed keys test
		attrib_table_ptr a3(new AttribTable());
		a3->setEntry(Imath::V3f(1,2,3), "Vek-tor!");
		__ASSERT( *a3->getEntry<std::string>(Imath::V3f(1,2,3)) == std::string("Vek-tor!") );
		__ASSERT( a3->getEntry<std::string>(Imath::V3f(1,1,3)) == NULL );
		a3->setEntry(Imath::M44f(2), Imath::M44d(4));
		__ASSERT( *a3->getEntry<Imath::M44d>(Imath::M44f(2)) == Imath::M44d(4) );
		__ASSERT( a3->getEntry<Imath::M44d>(Imath::M44f(3)) == NULL );


		a3->setEntry(Imath::V3f(1,3,3), "Vek-tor!");
		a3->setEntry(Imath::V3f(2,3,3), "Vek-tor!");
		a3->setEntry(Imath::V3f(1,3,4), "Vek-tor!");


		a3->setEntry(Imath::M33f(1), "Mf1!");
		a3->setEntry(Imath::M33d(2), "Md2!");
		a3->setEntry(Imath::M33d(1), "Md1!");

		a3->setEntry(1,2);
		a3->setEntry(0.5, 0.25f);
		a3->setEntry(1.5, 2.25f);
		a3->setEntry("str","ing");

		std::string strTemp;

		std::vector<int> vInt;
		std::vector<std::string> vStr;
		vInt.push_back(1);
		vInt.push_back(2);
		vInt.push_back(1);
		vStr.push_back("a");
		vStr.push_back("r");
		vStr.push_back("r");
		vStr.push_back("a");
		vStr.push_back("y");
		a3->setEntry("arr", vInt);
		a3->setEntry(vStr, 5);

		std::set<int> sInt;
		std::set<std::string> sStr;
		sInt.insert(1);
		sInt.insert(2);
		sInt.insert(1);
		sStr.insert("a");
		sStr.insert("r");
		sStr.insert("r");
		sStr.insert("a");
		sStr.insert("y");
		a3->setEntry("set", sInt);
		a3->setEntry(sStr, 5);


		object_rawptr_set printed;
		for( AttribTable::iterator it = a3->begin(); it != a3->end(); ++it )
		{
			it->first->str(std::cout, printed);
			std::cout << ":";
			it->second->str(std::cout, printed);
			std::cout << std::endl;
		}
		std::cout << *a3 << std::endl;
		a3->Object::dump();
		std::cout << std::endl;
	}

	std::cout<<std::endl;

	std::for_each( a->begin(), a->end(), printEntry );

	std::cout << "success" << std::endl;


	{
		attrib_ptr a(new TypedAttribute<int>(55));
		IntAttribPtr b(new TypedAttribute<int>(55));

		__ASSERT(*a == *b);
		b->value()++;
		__ASSERT(*a != *b);
	}

	object_ptr tk( new TypedAttribute<std::set<int, util::less<int>, util::counted_allocator<int> > >( 1, 2, 1, 2, 3, 4 ) );
	save(tk, "/tmp/tk.xml");
	object_ptr tk2( load("/tmp/tk.xml"));
	__ASSERT(!( *static_cast<const Attribute *>(tk.get()) < *static_cast<const Attribute *>(tk2.get() ) ));
	__ASSERT(!( *static_cast<const Attribute *>(tk2.get()) < *static_cast<const Attribute *>(tk.get() ) ));
	__ASSERT(*static_cast<const Attribute *>(tk2.get()) == *static_cast<const Attribute *>(tk.get() ) );

	std::vector<int> vec;
	vec.push_back(1);
	vec.push_back(2);
	vec.push_back(1);
	vec.push_back(2);
	vec.push_back(4);

	save( t, "/tmp/test.xml" );
	save( t, "/tmp/test.nap" );
	std::cout << "written" << std::endl;
	object_ptr obptr( load( "/tmp/test.xml" ) );
	object_ptr obptr2( load( "/tmp/test.nap" ) );
	std::cout << "original:" << std::endl;
	t->Object::dump();
	std::cout << std::endl;
	std::cout << "xml:" << std::endl;
	obptr->Object::dump();
	std::cout << std::endl;
	std::cout << "nap:" << std::endl;
	obptr2->Object::dump();
	std::cout << std::endl;

	{
		attrib_table_ptr extractTest(new AttribTable());
		extractTest->setEntry("float", 1.5f);
		extractTest->setEntry("double", 1.5);
		extractTest->setEntry("int", 1);
		extractTest->setEntry("unsigned int", (unsigned int)1);
		extractTest->setEntry("char", 'c');
		extractTest->setEntry("unsigned char", (unsigned char)'c');
		extractTest->setEntry("short", (short)1);
		extractTest->setEntry("unsigned short", (unsigned short)1);
		extractTest->setEntry("bool", false);
		extractTest->setEntry("V3f", Imath::V3f(1, 2, 3));
		extractTest->setEntry("V3d", Imath::V3d(2, 4, 6));
		extractTest->setEntry("M33f", Imath::M33f(3));
		extractTest->setEntry("M33d", Imath::M33d(5));
		extractTest->setEntry("String", "this is not a test");

		std::vector<float> vf;
		vf.push_back(2.5);
		extractTest->setEntry("[float]", vf);

		std::vector<double> vd;
		vd.push_back(2.5);
		extractTest->setEntry("[double]", vd);

		std::vector<int> vi;
		vi.push_back(2);
		extractTest->setEntry("[int]", vi);

		std::set<int> si;
		si.insert(4);
		extractTest->setEntry("set(int)", si);


		runExtractTest<float>(extractTest, "float");
		runExtractTest<double>(extractTest, "double");
		runExtractTest<int>(extractTest, "int");
		runExtractTest<unsigned int>(extractTest, "unsigned int");
		runExtractTest<char>(extractTest, "char");
		runExtractTest<unsigned char>(extractTest, "unsigned char");
		runExtractTest<short>(extractTest, "short");
		runExtractTest<unsigned short>(extractTest, "unsigned short");
		runExtractTest<bool>(extractTest, "bool");
		runExtractTest<Imath::V3f>(extractTest, "V3f");
		runExtractTest<Imath::V3d>(extractTest, "V3d");
		runExtractTest<Imath::M33f>(extractTest, "M33f");
		runExtractTest<Imath::M33d>(extractTest, "M33d");
		runExtractTest<std::string>(extractTest, "String");
		runExtractTest<std::vector<float, util::counted_allocator<float> > >(extractTest, "[float]");
		runExtractTest<std::vector<double, util::counted_allocator<double> > >(extractTest, "[double]");
		runExtractTest<std::vector<int, util::counted_allocator<int> > >(extractTest, "[int]");
		runExtractTest<std::set<int, util::less<int>, util::counted_allocator<int> > >(extractTest, "set(int)");
	}

	{
		// Test serialization of all attribute types.
		attrib_table_ptr uberTable(new AttribTable());
#define _NAPALM_TYPE_OP(T, Label) uberTable->setEntry(#Label, util::default_construction<T>::value() );\
		{\
			std::set<T, util::less<T> > s; s.insert(util::default_construction<T>::value());\
			uberTable->setEntry(std::string("Set")+#Label, s);\
			std::vector<T> v; v.push_back(util::default_construction<T>::value());\
			uberTable->setEntry(std::string("Vector")+#Label, v);\
		}
#include "../napalm/core/types/all.inc"
#undef _NAPALM_TYPE_OP

		uberTable->setEntry( "boom", false );
		napalm::AttributeList l;
		l.push_back("This");
		l.push_back(1);
		l.push_back(true);
		uberTable->setEntry( "liszt", l );

		save( uberTable, "/tmp/test.xml" );
		save( uberTable, "/tmp/test.nap" );
		object_ptr uber1( load( "/tmp/test.xml" ) );
		napalm::Table<napalm::Attribute> & t1 = *(napalm::Table<napalm::Attribute>*)uber1.get();
		object_ptr uber2( load( "/tmp/test.nap" ) );
		napalm::Table<napalm::Attribute> & t2 = *(napalm::Table<napalm::Attribute>*)uber2.get();
		napalm::Table<napalm::Attribute>::const_iterator it = uberTable->begin(), it1 = t1.begin(), it2 = t2.begin();
		for( ;it != uberTable->end(); ++it, ++it1, ++it2)
		{
			if( *it->second != *it1->second )
			{
				std::cout << *it->second << " != "<<*it1->second<<std::endl;
			}
			if( *it->second != *it2->second )
			{
				std::cout << *it->second << " != "<<*it2->second<<std::endl;
			}
		}
		__ASSERT( it1 == t1.end() );
		__ASSERT( it2 == t2.end() );
		__ASSERT( areEqual( uber1, uberTable ));
		__ASSERT( areEqual( uber2, uberTable ));
	}

	std::cout << "\nAll tests passed, huzzah!\n" << std::endl;

	return 0;
}






