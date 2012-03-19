

#include "adaptors/points.hpp"
#include "adaptors/mesh.hpp"
#include <maya/MFloatPointArray.h>

#include <maya/MItMeshPolygon.h>
#include <maya/MFnMesh.h>
#include <assert.h>

namespace dgal {

	/*
	 * @brief Maya point specialization
	 */
	template<>
	struct points_adaptor<MFloatPointArray>
	{
		typedef Imath::V3f			elem_type;
		typedef Imath::V3f			const_elem_ref;
		typedef MFloatPointArray	target_type;
		typedef float				scalar;

		points_adaptor(const target_type& t):m_target(t){}
		inline unsigned int size() const { return m_target.length(); }
		inline unsigned int index(unsigned int i) const { return i; }

		inline const_elem_ref operator[](unsigned int i) const {
			const MFloatPoint& p = m_target[i];
			return const_elem_ref(p.x, p.y, p.z);
		}

		const target_type& m_target;
	};

	/*
	 * @brief GEO_PrimPoly points specialization
	 */
	template<>
	struct points_adaptor<MItMeshPolygon>
	{
		typedef Imath::V3f		elem_type;
		typedef Imath::V3f		const_elem_ref;
		typedef MItMeshPolygon	target_type;
		typedef float			scalar;

		points_adaptor(const target_type& t):m_target(t){}
		inline unsigned int size() const { return (const_cast<MItMeshPolygon*>(&m_target))->polygonVertexCount(); }
		inline bool is_indexed() const { return true; }

		inline unsigned int index(unsigned int i) const
		{
			MStatus stat;
			unsigned ind= (const_cast<MItMeshPolygon*>(&m_target))->vertexIndex(i,&stat); //global mesh index
			if (!stat)
			{
				stat.perror("point_adaptor unsigned index(unsigned int i");
			}

		}

		inline const_elem_ref operator[](unsigned int i) const
		{
			MStatus stat;
			const MPoint& p = (const_cast<MItMeshPolygon*>(&m_target))->point(i,MSpace::kObject,&stat);
			if (!stat)
			{
				stat.perror("points_adaptor  const_elem_ref operator[]");
			}
			return const_elem_ref(p.x, p.y, p.z);
		}

		const target_type& m_target;
	};

	/*
	 * @brief Maya mesh specialization
	 */
	template<>
	struct mesh_adaptor<MObject>
	{
		typedef MObject							target_type;
		typedef float							scalar;
		typedef unsigned int					index_type;
		typedef Imath::V3f						point_type;
		typedef Imath::V3f						const_point_ref;
		typedef MItMeshPolygon					poly_type;
		typedef const MItMeshPolygon&			const_poly_ref;

		mesh_adaptor(const target_type& t ):m_target(t), m_ItPoly(t),m_mFn(t)
		{

		}

		inline unsigned int numPoints() const { return m_mFn.numVertices(); }
		inline unsigned int numPolys() const { return m_mFn.numPolygons(); }

		inline const_point_ref getPoint(unsigned int i) const
		{
			MStatus stat;
			assert(i < m_mFn.numVertices());
			MPoint p;
			stat = m_mFn.getPoint(i,p,MSpace::kObject);
			if(!stat)
			{
				stat.perror("const_point_ref getPoint(unsigned int i)");
			}

			return const_point_ref(p.x, p.y, p.z);
		}


		inline const_poly_ref getPoly(unsigned int i)
		{
			MStatus stat;
			int prevIndex;

			stat = m_ItPoly.setIndex(i,prevIndex);
			if (!stat)
			{
				stat.perror("meshAdaptor::getPoly, mIt.setIndex(i,previndex)");
			}

			return m_ItPoly;
		}

		const target_type& m_target;
		MFnMesh m_mFn;
		MItMeshPolygon m_ItPoly;
	};


}
























