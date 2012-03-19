#include <hdk_utils/ScopedCook.h>
#include <hdk_utils/util.h>
#include <hdk_utils/OP_Params.h>
#include <hdk_utils/GeoAttributeCopier.h>
#include <UT/UT_DSOVersion.h>
#include <OP/OP_OperatorTable.h>
#include <GU/GU_PrimPoly.h>
#include <houdini/PRM/PRM_Include.h>
#include <dgal/adaptors/houdini.hpp>
#include <dgal/MeshConnectivity.hpp>
#include <dgal/algorithm/addMeshEdges.hpp>
#include <dgal/algorithm/remapMesh.hpp>
#include "SOP_VoroUVProject.h"
#include "util/simple_mesh.h"


using namespace clip_sops;
using namespace hdk_utils;


void newSopOperator(OP_OperatorTable *table)
{
    table->addOperator(
	    new OP_Operator(SOP_NAME,			// Internal name
			 SOP_LABEL,						// UI name
			 SOP_VoroUVProject::myConstructor,		// How to build the SOP
			 SOP_VoroUVProject::myTemplateList,		// My parameters
			 0,								// Min # of sources
			 1,								// Max # of sources
			 SOP_VoroUVProject::myVariables,		// Local variables
			 0)
	    );
}


//static PRM_Name VUV_ParmNameCellIDAttrib( "cell_id_attrib", "Cell ID" );
//static PRM_Default VUV_ParmDefaultCellIDAttrib(0, "");

static PRM_Name VUV_ParmNameUVAttrib( "uv_attrib", "UV" );
static PRM_Default VUV_ParmDefaultUVAttrib(0, "uv");

static PRM_Name VUV_ParmNameUVRegionAttrib( "uv_region_attrib", "UV Region" );
static PRM_Default VUV_ParmDefaultUVRegionAttrib(0, "");

static PRM_Name VUV_ParmNameFaceAttrib( "face_id_attrib", "Face ID" );
static PRM_Default VUV_ParmDefaultFaceAttrib(0, "");

PRM_Template SOP_VoroUVProject::myTemplateList[] = {
	PRM_Template(PRM_STRING, 1, &VUV_ParmNameUVAttrib, &VUV_ParmDefaultUVAttrib),
	PRM_Template(PRM_STRING, 1, &VUV_ParmNameUVRegionAttrib, &VUV_ParmDefaultUVRegionAttrib),
	PRM_Template(PRM_STRING, 1, &VUV_ParmNameFaceAttrib, &VUV_ParmDefaultFaceAttrib),
	PRM_Template()
};

CH_LocalVariable SOP_VoroUVProject::myVariables[] = {
	{ 0 }		// End the table with a null entry
};


const char* SOP_VoroUVProject::inputLabel(unsigned int idx) const
{
	return "Voronoi-fractured Mesh";
}


OP_Node* SOP_VoroUVProject::myConstructor(OP_Network *net, const char *name, OP_Operator *op)
{
    return new SOP_VoroUVProject(net, name, op);
}


SOP_VoroUVProject::SOP_VoroUVProject(OP_Network *net, const char *name, OP_Operator *op)
:	SOP_Node(net, name, op)
{
}


float SOP_VoroUVProject::getVariableValue(int index, int thread)
{
    return SOP_Node::getVariableValue(index, thread);
}


OP_ERROR SOP_VoroUVProject::cookMySop(OP_Context &context)
{
	typedef dgal::simple_mesh<Imath::V3f>				simple_mesh;
	typedef dgal::MeshConnectivity<unsigned int>		mesh_conn;
	typedef mesh_conn::index_set						conn_set;
	typedef std::pair<unsigned int, unsigned int>		edge_type;
	typedef std::pair<unsigned int, unsigned int>		vertex_type; // face:vert
	typedef Imath::V2f									uv_type;
	typedef boost::unordered_map<vertex_type,uv_type>	vertex_uv_map;

	hdk_utils::ScopedCook scc(*this, context, "Performing drd voro uv projection");
	if(!scc.good())
		return error();


	////////////////////////////////////////////////////////////////////////////
	// get inputs
	////////////////////////////////////////////////////////////////////////////

	const GU_Detail* gdp0 = inputGeo(0);
	if (!gdp0 ) {
		SOP_ADD_FATAL(SOP_MESSAGE, "Not enough sources specified.");
	}

	std::string uvAttribStr = 		hdk_utils::getStringParam(*this, context, "uv_attrib");
	std::string uvRegionAttribStr = hdk_utils::getStringParam(*this, context, "uv_region_attrib");
	std::string faceAttribStr = 	hdk_utils::getStringParam(*this, context, "face_id_attrib");

	if(uvAttribStr.empty()) {
		SOP_ADD_FATAL(SOP_MESSAGE, "UV attribute unspecified.");
	}

	if(uvRegionAttribStr.empty()) {
		SOP_ADD_FATAL(SOP_MESSAGE, "UV region attribute unspecified.");
	}

	if(faceAttribStr.empty()) {
		SOP_ADD_FATAL(SOP_MESSAGE, "Face ID attribute unspecified.");
	}


	////////////////////////////////////////////////////////////////////////////
	// validate inputs
	////////////////////////////////////////////////////////////////////////////

	GEO_AttributeHandle hUV = gdp0->getAttribute(GEO_VERTEX_DICT, uvAttribStr.c_str());
	if(!hUV.isAttributeValid()) {
		SOP_ADD_FATAL(SOP_MESSAGE, "Must have vertex uv attribute.");
	}

	GEO_AttributeHandle hRegionID = gdp0->getAttribute(GEO_PRIMITIVE_DICT, uvRegionAttribStr.c_str());
	if(!hRegionID.isAttributeValid()) {
		SOP_ADD_FATAL(SOP_MESSAGE, "Must have primitive uv-region attribute.");
	}

	GEO_AttributeHandle hFaceID = gdp0->getAttribute(GEO_PRIMITIVE_DICT, faceAttribStr.c_str());
	if(!hFaceID.isAttributeValid()) {
		SOP_ADD_FATAL(SOP_MESSAGE, "Must have primitive face-id attribute.");
	}

	if(hRegionID.getAttribute()->getType() != GB_ATTRIB_INT) {
		SOP_ADD_FATAL(SOP_MESSAGE, "uv-region attribute must be INT type.");
	}

	if(hFaceID.getAttribute()->getType() != GB_ATTRIB_INT) {
		SOP_ADD_FATAL(SOP_MESSAGE, "face-id attribute must be INT type.");
	}

	if((hUV.getAttribute()->getType() != GB_ATTRIB_FLOAT) ||
		(hUV.getAttribute()->getSize() < 2)) {
		SOP_ADD_FATAL(SOP_MESSAGE, "uv attribute must be FLOAT[2+] type.");
	}

	const GEO_PrimList& prims0 = gdp0->primitives();
	const GEO_PointList& points0 = gdp0->points();


	////////////////////////////////////////////////////////////////////////////
	// calculate splitting edges
	////////////////////////////////////////////////////////////////////////////

	std::vector<int> polys_remap;
	simple_mesh split_mesh;

	{
		std::vector<edge_type> new_edges;

		mesh_conn conn;
		conn.set<GEO_Detail>(*gdp0, true);

		for(unsigned int i=0; i<prims0.entries(); ++i)
		{
			const GEO_PrimPoly* poly = dynamic_cast<const GEO_PrimPoly*>(prims0[i]);
			if(!poly)
				continue;

			hFaceID.setElement(poly);
			int val = hFaceID.getI();
			if(val != 0)
				continue;

			unsigned int nverts = poly->getVertexCount();
			std::vector<int> vert_types(nverts, 0);
			unsigned int n_inner_v = 0;
			unsigned int n_uvseam_v = 0;
			UT_Vector4 uvseam_vert, uvseam_tangent;

			// categorise verts into: inner, on-uv-seam
			for(unsigned int j=0; j<nverts; ++j)
			{
				const GEO_Point* ppt = poly->getVertex(j).getPt();
				int ptnum = ppt->getNum();
				const conn_set& pt_faces = conn.m_pt_face[ptnum];

				std::set<int> uv_regions;
				for(conn_set::const_iterator it=pt_faces.begin(); it!=pt_faces.end(); ++it)
				{
					hFaceID.setElement(prims0[*it]);
					val = hFaceID.getI();
					if(val != 0)
					{
						hRegionID.setElement(prims0[*it]);
						uv_regions.insert(hRegionID.getI());
					}
				}

				if(uv_regions.empty())
				{
					vert_types[j] = 2;
					++n_inner_v;
				}
				else if(uv_regions.size() > 1)
				{
					vert_types[j] = 1;
					if(n_uvseam_v == 0)
					{
						uvseam_vert = ppt->getPos();

						// store a 'tangent' for the first uvseam vert, use this to select the
						// 'best' inner vert to create an edge to
						const GEO_Point* ppt1 = poly->getVertex((j+nverts-1)%nverts).getPt();
						const GEO_Point* ppt2 = poly->getVertex((j+1)%nverts).getPt();
						uvseam_tangent = ppt->getPos() - ppt1->getPos();
						uvseam_tangent.normalize();
					}

					++n_uvseam_v;
				}
			}

			// note: we give up if there is a uv seam but no inner verts, should be rare
			if((n_uvseam_v > 0) && (n_inner_v > 0))
			{
				// find inner vert most opposite uvseam vert on the poly
				float mindot = 2.f;
				int target_ptnum = -1;
				for(unsigned int j=0; j<nverts; ++j)
				{
					if(vert_types[j] == 2)
					{
						const GEO_Point* ppt = poly->getVertex(j).getPt();
						UT_Vector4 to_v = ppt->getPos() - uvseam_vert;
						to_v.normalize();
						float f = std::abs(to_v.dot(uvseam_tangent));
						if(f < mindot)
						{
							mindot = f;
							target_ptnum = ppt->getNum();
						}
					}
				}

				// create new edge(s)
				for(unsigned int j=0; j<nverts; ++j)
				{
					if(vert_types[j] == 1)
					{
						const GEO_Point* ppt = poly->getVertex(j).getPt();
						unsigned int seam_ptnum = static_cast<unsigned int>(ppt->getNum());
						new_edges.push_back(edge_type(seam_ptnum, target_ptnum));
					}
				}
			}
		}

		// apply splitting edges
		dgal::addMeshEdges<GEO_Detail, std::vector<edge_type>::const_iterator>(
			*gdp0, split_mesh, new_edges.begin(), new_edges.end(), &polys_remap);
	}


	////////////////////////////////////////////////////////////////////////////
	// bleed uvs onto inner faces connected to surface faces
	////////////////////////////////////////////////////////////////////////////

	// (uv-region -> ((face:vertex) -> u,v))
	std::map<int, vertex_uv_map> vertex_uvs_by_uvregion;
	//vertex_uv_map vertex_uvs;

	// (uvregion -> bleed faces)
	typedef boost::unordered_map<int, conn_set> bleed_faces_map;
	bleed_faces_map bleed_faces;

	{
		// find bleed faces and group into uv regions
		mesh_conn conn;
		conn.set<simple_mesh>(split_mesh, false, true, false, true);

		for(mesh_conn::face_face_map::const_iterator it=conn.m_face_face.begin();
			it!=conn.m_face_face.end(); ++it)
		{
			int orig_face_id = std::abs(polys_remap[it->first]) - 1;
			hFaceID.setElement(prims0[orig_face_id]);
			if(hFaceID.getI() == 0)
			{
				const conn_set& faces = it->second;
				int region_id = -1;

				for(conn_set::const_iterator it2=faces.begin(); it2!=faces.end(); ++it2)
				{
					orig_face_id = std::abs(polys_remap[*it2]) - 1;
					hFaceID.setElement(prims0[orig_face_id]);
					if(hFaceID.getI() != 0)
					{
						hRegionID.setElement(prims0[orig_face_id]);
						region_id = hRegionID.getI();
						break;
					}
				}

				if(region_id != -1)
					bleed_faces[region_id].insert(it->first);
			}
		}

		// bleed uvs onto bleed faces
		for(bleed_faces_map::const_iterator it=bleed_faces.begin(); it!=bleed_faces.end(); ++it)
		{
			int uv_region = it->first;
			vertex_uv_map& vertex_uvs = vertex_uvs_by_uvregion[uv_region];

			for(conn_set::const_iterator it2=it->second.begin(); it2!=it->second.end(); ++it2)
			{
				unsigned int face_id = *it2;
				const simple_mesh::polygon_type& poly = split_mesh.m_polys[face_id];
				unsigned int nverts = poly.size();

				// bleed onto verts that are incident to a surface pt
				unsigned int nbledverts = 0;
				for(unsigned int i=0; i<nverts; ++i)
				{
					unsigned int ptnum = poly[i];
					const mesh_conn::vertex_set& verts = conn.m_pt_vert[ptnum];

					for(mesh_conn::vertex_set::const_iterator it3=verts.begin(); it3!=verts.end(); ++it3)
					{
						if(it3->first == face_id)
							continue;

						int orig_face_id = std::abs(polys_remap[it3->first]) - 1;

						const GEO_PrimPoly* poly = dynamic_cast<const GEO_PrimPoly*>(prims0[orig_face_id]);
						if(!poly)
							continue;

						hFaceID.setElement(poly);
						if(hFaceID.getI() != 0)
						{
							hRegionID.setElement(poly);
							if(hRegionID.getI() == uv_region)
							{
								// found incident surface vert: get its uv
								assert(it3->second < poly->getVertexCount());
								const GEO_Vertex& vert = poly->getVertex(it3->second);
								hUV.setElement(&vert);
								uv_type uv;
								uv.x = hUV.getF(0);
								uv.y = hUV.getF(1);

								vertex_uvs[vertex_type(face_id,i)] = uv;
								++nbledverts;
								break;
							}
						}
					}
				}

				// for non-surface-incident verts, interpolate
				if(nbledverts > 0)
				{
					vertex_uv_map interp_vertices;

					for(unsigned int i=0; i<nverts; ++i)
					{
						unsigned int ptnum = poly[i];
						if(vertex_uvs.find(vertex_type(face_id,i)) != vertex_uvs.end())
							continue;

						float dist_back=0.f, dist_forward=0.f;
						uv_type uv_back, uv_forward;

						// find distance back to last bled vert
						unsigned int j = i+nverts-1;
						Imath::V3f pt = split_mesh.m_points[ptnum];
						while(1)
						{
							unsigned int next_ptnum = poly[j%nverts];
							const Imath::V3f& next_pt = split_mesh.m_points[next_ptnum];
							dist_back += (next_pt-pt).length();

							vertex_uv_map::const_iterator it3 =
								vertex_uvs.find(vertex_type(face_id, j%nverts));

							if(it3 != vertex_uvs.end())
							{
								uv_back = it3->second;
								break;
							}

							pt = next_pt;
							--j;
						}

						// find distance forward to next bled vert
						j = i+1;
						pt = split_mesh.m_points[ptnum];
						while(1)
						{
							unsigned int next_ptnum = poly[j%nverts];
							const Imath::V3f& next_pt = split_mesh.m_points[next_ptnum];
							dist_forward += (next_pt-pt).length();

							vertex_uv_map::const_iterator it3 =
								vertex_uvs.find(vertex_type(face_id, j%nverts));

							if(it3 != vertex_uvs.end())
							{
								uv_forward = it3->second;
								break;
							}

							pt = next_pt;
							++j;
						}

						// interp uvs
						float f = dist_back + dist_forward;
						uv_type uv_interp;

						if(f < std::numeric_limits<float>::epsilon())
							uv_interp = uv_back;
						else
							uv_interp = uv_back*(dist_forward/f) + uv_forward*(dist_back/f);

						interp_vertices[vertex_type(face_id,i)] = uv_interp;
					}

					vertex_uvs.insert(interp_vertices.begin(), interp_vertices.end());
				}
			}
		}
	}


	////////////////////////////////////////////////////////////////////////////
	// create gdp and remap attributes
	////////////////////////////////////////////////////////////////////////////

	// calculate attrib remapping
	dgal::MeshRemapSettings<int> remap_settings(false, true, true, 0, 0,
		NULL, &polys_remap, true);

	dgal::MeshRemapResult<unsigned int, float> remap_result;
	dgal::remapMesh<GEO_Detail, simple_mesh, int>(*gdp0, split_mesh,
		remap_settings, remap_result);

	// set gdp to new mesh
	gdp->clearAndDestroy();
	util::add_simple_mesh(*gdp, split_mesh);

	// remap attribs
	GeoAttributeCopier gc(*gdp);
	gc.add(*gdp0, GEO_DETAIL_DICT, GEO_DETAIL_DICT);
	gc.add(*gdp0, GEO_POINT_DICT, GEO_POINT_DICT);
	gc.add(*gdp0, GEO_PRIMITIVE_DICT, GEO_PRIMITIVE_DICT, &(remap_result.m_polyMapping));
	gc.add(*gdp0, GEO_VERTEX_DICT, GEO_VERTEX_DICT, &(remap_result.m_vertexMapping));
	gc.add(*gdp0, GEO_VERTEX_DICT, GEO_VERTEX_DICT, &(remap_result.m_vertexIMapping),
		GeoAttributeCopier::GAC_MERGE);

	gc.apply();


	////////////////////////////////////////////////////////////////////////////
	// write calculated uvs onto bleed vertices
	////////////////////////////////////////////////////////////////////////////

	{
		GEO_PrimList& prims = gdp->primitives();

		GEO_AttributeHandle hUV2 = gdp->getAttribute(GEO_VERTEX_DICT, uvAttribStr.c_str());
		if(!hUV2.isAttributeValid()) {
			SOP_ADD_FATAL(SOP_MESSAGE, "Internal error #1 see rnd!");
		}

		for(std::map<int, vertex_uv_map>::const_iterator it2=vertex_uvs_by_uvregion.begin();
			it2!=vertex_uvs_by_uvregion.end(); ++it2)
		{
			const vertex_uv_map& vertex_uvs = it2->second;

			for(vertex_uv_map::const_iterator it=vertex_uvs.begin(); it!=vertex_uvs.end(); ++it)
			{
				unsigned int face_id = it->first.first;
				unsigned int vert_index = it->first.second;
				const uv_type& uv = it->second;

				GEO_PrimPoly* poly = dynamic_cast<GEO_PrimPoly*>(prims[face_id]);
				if(!poly)
					continue;

				GEO_Vertex& vert = poly->getVertex(vert_index);

				hUV2.setElement(&vert);
				hUV2.setF(uv.x, 0);
				hUV2.setF(uv.y, 1);
			}
		}
	}

	return error();
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
