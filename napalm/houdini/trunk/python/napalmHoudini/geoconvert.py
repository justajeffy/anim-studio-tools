import hou

import napalm
import pimath


DETAIL_CATEGORY = 'Global'
PRIMITIVE_CATEGORY = 'Primitive'
POINT_CATEGORY = 'Point'
VERTEX_CATEGORY = 'Vertex'
GEOMETRY_CATEGORY = 'Geometry'


STRING_DATA_TYPE = 'String'
FLOAT_DATA_TYPE = 'Float'
INT_DATA_TYPE = 'Int'


CAST = {}
CAST[STRING_DATA_TYPE] = ['str', 'str', 'str', 'str']
CAST[FLOAT_DATA_TYPE] = ['float', 'pimath.V2f', 'pimath.V3f', 'pimath.V4f']
CAST[INT_DATA_TYPE] = ['int', 'pimath.V2i', 'pimath.V3i', 'pimath.V4i']


BUFFER_TYPE = {}
BUFFER_TYPE[STRING_DATA_TYPE] = ['napalm.StringBuffer',
                                 None,
                                 None,
                                 None]
BUFFER_TYPE[FLOAT_DATA_TYPE] = ['napalm.FloatBuffer',
                                'napalm.V2fBuffer',
                                'napalm.V3fBuffer',
                                'napalm.V4fBuffer']
BUFFER_TYPE[INT_DATA_TYPE] = ['napalm.IntBuffer',
                              'napalm.V2iBuffer',
                              'napalm.V3iBuffer',
                              'napalm.V4iBuffer']


def attribute_to_buffer(attribute, hou_class_tuple):
    """hou_class_tuple must be a tuple of Primitive, Point or Vertex instances.
    """
    name = attribute.name()
    #~print name
    buffer_size = len(hou_class_tuple)
    category = None
    if buffer_size:
        try:
            first_item = hou_class_tuple[0]
            category = first_item.attribType().name()
        except:
            raise Exception("Unable to determine attribute type for: %s"
                            % first_item)
    #~print category
    data_type = attribute.dataType().name()
    #~print data_type
    dimensions = attribute.size()
    #~print dimensions

    if dimensions == 1:
        pass
    elif dimensions == 2:
        if data_type == STRING_DATA_TYPE:
            raise Exception("Strings with %s dimensions are not supported"
                            % dimensions)
    elif dimensions == 3:
        if data_type == STRING_DATA_TYPE:
            raise Exception("Strings with %s dimensions are not supported"
                            % dimensions)
    elif dimensions == 4:
        if data_type == STRING_DATA_TYPE:
            raise Exception("Strings with %s dimensions are not supported"
                            % dimensions)
    else:
        raise Exception("Attributes with %s dimensions are not supported: %s"
                        % (dimensions, attr))

    # Create buffer
    buffer = None
    statement = '%s(%s)' % (BUFFER_TYPE[data_type][dimensions-1],
                            buffer_size)
    #~print statement
    buffer = eval(statement)
    # Write values
    i = 0
    for item in hou_class_tuple:
        value = eval('%s(item.attribValue(attribute))' %
                     CAST[data_type][dimensions-1])
        buffer.w(i, value)
        i += 1

    # Buffer attributes
    buffer.attribs['name'] = name
    buffer.attribs['dimensions'] = dimensions
    buffer.attribs['category'] = category
    buffer.attribs['dataType'] = data_type
    buffer.attribs['isTransformedAsNormal'] = attribute.isTransformedAsNormal()
    buffer.attribs['defaultValue'] = eval('%s(attribute.defaultValue())' %
                                          CAST[data_type][dimensions-1])

    return buffer


def geo_to_table(geometry):
    geo = geometry
    table = napalm.ObjectTable()

    # Detail attributes
    detail_attributes = {}
    for a in geo.globalAttribs():
        detail_attributes[a.name()] = a
    #~print 'Detail attributes: %s' % detail_attributes

    for name, attr in detail_attributes.items():
        attrib_table = napalm.AttribTable()
        attrib_table['name'] = attr.name()
        size = attr.size()
        attrib_table['dimensions'] = size
        attrib_table['category'] = attr.type().name()
        data_type = attr.dataType().name()
        attrib_table['dataType'] = data_type
        attrib_table['isTransformedAsNormal'] = attr.isTransformedAsNormal()
        attrib_table['defaultValue'] = eval('%s(attr.defaultValue())'
                                            % CAST[data_type][size-1])
        attrib_table['value'] = eval('%s(geo.attribValue(attr))'
                                     % CAST[data_type][size-1])
        table['%s:%s' % (DETAIL_CATEGORY, name)] = attrib_table

    # Primitives
    prims = geo.prims()
    npr = len(prims)
    #~print 'Number of prims: %s' % npr
    primitive_attributes = {}
    for a in geo.primAttribs():
        primitive_attributes[a.name()] = a
    #~print 'Primitve attributes: %s' % primitive_attributes

    for name, attr in primitive_attributes.items():
        buffer = attribute_to_buffer(attr, prims)
        table['%s:%s' % (PRIMITIVE_CATEGORY, name)] = buffer

    # Points
    points = geo.points()
    npt = len(points)
    #~print 'Number of points: %s' % npt
    point_attributes = {}
    for a in geo.pointAttribs():
        # Special case for P and Pw
        if a.name() == 'P':
            point_attributes['P'] = a
        elif a.name() == 'Pw':
            point_attributes['Pw'] = a
        else:
            point_attributes[a.name()] = a
    #~print 'Point attributes: %s' % point_attributes

    for name, attr in point_attributes.items():
        buffer = attribute_to_buffer(attr, points)
        table['%s:%s' % (POINT_CATEGORY, name)] = buffer

    # Vertices, dependent on number of primitives (npr) defined above
    vertices = geo.globVertices('0-%s' % (npr-1))
    nvtx = len(vertices)
    #~print 'Number of verts: %s' % nvtx
    vertex_attributes = {}
    for a in geo.vertexAttribs():
        vertex_attributes[a.name()] = a
    #~print 'Vertex attributes: %s' % vertex_attributes

    for name, attr in vertex_attributes.items():
        buffer = attribute_to_buffer(attr, vertices)
        table['%s:%s' % (VERTEX_CATEGORY, name)] = buffer

    # Geometry information
    prim_types = napalm.StringBuffer(npr)
    verts_per_prim = napalm.IntBuffer(npr)
    i = 0
    for item in prims:
        prim_types.w(i, item.type().name())
        verts_per_prim.w(i, len(item.vertices()))
        i += 1
    table['%s:%s' % (GEOMETRY_CATEGORY, 'primitiveTypes')] = prim_types
    table['%s:%s' % (GEOMETRY_CATEGORY, 'verticesPerPrimitive')] = verts_per_prim

    vert_indexes = napalm.IntBuffer(nvtx)
    i = 0
    for item in vertices:
        vert_indexes.w(i, item.point().number())
        i += 1

    table['%s:%s' % (GEOMETRY_CATEGORY, 'vertices')] = vert_indexes
    table['%s:%s' % (GEOMETRY_CATEGORY, 'numberOfPoints')] = npt
    table['%s:%s' % (GEOMETRY_CATEGORY, 'numberOfPrimitives')] = npr
    table['%s:%s' % (GEOMETRY_CATEGORY, 'numberOfVertices')] = nvtx

    return table


def create_geometry(table, geometry):
    npt = table['%s:numberOfPoints' % GEOMETRY_CATEGORY]
    prim_types = table['%s:primitiveTypes' % GEOMETRY_CATEGORY]
    verts_per_prim = table['%s:verticesPerPrimitive' % GEOMETRY_CATEGORY]
    npr = table['%s:numberOfPrimitives' % GEOMETRY_CATEGORY]
    vertices = table['%s:vertices' % GEOMETRY_CATEGORY]

    points = []
    for n in range(npt):
        points.append(geometry.createPoint())

    prims = []
    vertices_index = 0
    for i in range(npr):
        prim_type = prim_types.r(i)
        num_verts = verts_per_prim.r(i)

        creator = 'geometry.create%s()' % prim_type
        prim = eval(creator)

        for v in range(num_verts):
            point_number = vertices.r(vertices_index)
            prim.addVertex(points[point_number])
            vertices_index += 1
        prims.append(prim)

    return geometry


def get_attribute_buffers(category, table, geometry):
    buffers = {}
    for key in table.keys():
        if key.startswith('%s:' % category):
            buffer = table[key]
            attrib_name = buffer.attribs['name']
            buffers[attrib_name] = buffer
    return buffers


def get_attribute_tables(category, table, geometry):
    attribute_tables = {}
    for key in table.keys():
        if key.startswith('%s:' % category):
            attrib_table = table[key]
            attrib_name = table[key]['name']
            attribute_tables[attrib_name] = attrib_table
    return attribute_tables


def pimath_to_tuple(pimath_object, dimensions):
    # Convert from pimath type to a standard tuple
    value_list = []
    for dim in ['x', 'y', 'z', 'w']:
        if len(value_list) < dimensions:
            value_list.append(eval('pimath_object.%s' % dim))
        else:
            break
    value_tuple = tuple(value_list)
    return value_tuple


def buffer_to_attribute(buffer, hou_class_tuple, geometry):
    category = buffer.attribs['category']
    name = buffer.attribs['name']

    # Special case for P and Pw point attributes
    if category == POINT_CATEGORY:
        if name == 'P':
            return geometry.findPointAttrib('P')
        elif name == 'Pw':
            return geometry.findPointAttrib('Pw')
        else:
            pass

    data_type = buffer.attribs['dataType']
    transform_as_normal = buffer.attribs['isTransformedAsNormal']

    default_value = buffer.attribs['defaultValue']
    dimensions = buffer.attribs['dimensions']

    if dimensions > 1:
        default_value = pimath_to_tuple(default_value, dimensions)

    attrib_type = None
    try:
        attrib_type = eval('hou.attribType.%s' % category)
    except:
        raise Exception('Unknown attribType: %s' % category)

    attribute = geometry.addAttrib(attrib_type, name, default_value,
                                   transform_as_normal)
    return attribute


def attrib_table_to_attribute(attrib_table, geometry):
    print attrib_table['name']
    print attrib_table['dimensions']
    print attrib_table['dataType']
    print attrib_table['category']
    print attrib_table['defaultValue']


def table_to_geo(table, geometry):
    geo = geometry
    geo.clear()

    # Create points
    geo = create_geometry(table, geo)
    points = geo.points()
    npt = len(points)

    point_attribute_buffers = get_attribute_buffers(POINT_CATEGORY, table,
                                                    geometry)
    point_attributes = {}

    for name, buffer in point_attribute_buffers.items():
        point_attributes[name] = buffer_to_attribute(buffer, points, geo)

    #~print point_attributes

    for name, attr in point_attributes.items():
        for i in range(npt):
            buffer = point_attribute_buffers[name]
            dimensions = buffer.attribs['dimensions']
            value = buffer[i]
            if dimensions > 1:
                value = pimath_to_tuple(value, dimensions)
            points[i].setAttribValue(attr, value)

    # Create primitives
    prims = geo.prims()
    npr = len(prims)

    primitive_attribute_buffers = get_attribute_buffers(PRIMITIVE_CATEGORY,
                                                        table, geometry)

    primitive_attributes = {}

    for name, buffer in primitive_attribute_buffers.items():
        primitive_attributes[name] = buffer_to_attribute(buffer, prims, geo)

    #~print primitive_attributes

    for name, attr in primitive_attributes.items():
        for i in range(npr):
            buffer = primitive_attribute_buffers[name]
            dimensions = buffer.attribs['dimensions']
            value = buffer[i]
            if dimensions > 1:
                value = pimath_to_tuple(value, dimensions)
            prims[i].setAttribValue(attr, value)

    # Vertices, dependent on number of primitives (npr) defined above
    vertices = geo.globVertices('0-%s' % (npr-1))
    nvtx = len(vertices)

    vertex_attribute_buffers = get_attribute_buffers(VERTEX_CATEGORY,
                                                     table, geometry)

    vertex_attributes = {}

    for name, buffer in vertex_attribute_buffers.items():
        vertex_attributes[name] = buffer_to_attribute(buffer, vertices, geo)

    #~print vertex_attributes

    for name, attr in vertex_attributes.items():
        for i in range(nvtx):
            buffer = vertex_attribute_buffers[name]
            dimensions = buffer.attribs['dimensions']
            value = buffer[i]
            if dimensions > 1:
                value = pimath_to_tuple(value, dimensions)
            vertices[i].setAttribValue(attr, value)

    return geo

# Copyright 2008-2012 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios)
#
# This file is part of anim-studio-tools.
#
# anim-studio-tools is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# anim-studio-tools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with anim-studio-tools.  If not, see <http://www.gnu.org/licenses/>.

