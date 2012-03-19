#version 150 core

// VertexShader Uniforms
uniform mat4 ModelMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ProjMatrix;

// Input from VBOs
in vec3 iPosition;
in vec3 iNormal;
in vec2 iTexCoord;

// Output to FragmentShader
out vec3 oPosition;
out vec3 oNormal;
out vec2 oTexCoord;
out vec2 oOrgPosition;

void main(void)
{
	mat4 mvpMatrix = ProjMatrix * ViewMatrix * ModelMatrix;
	gl_Position = mvpMatrix * vec4(iPosition, 1);

	mat4 mvMatrix = ViewMatrix * ModelMatrix;
	vec4 viewPos = mvMatrix * vec4(iPosition, 1);
	oPosition = viewPos.xyz;
	
	//oPosition = gl_Position.xyz; // ssao test
	oOrgPosition = iPosition.xz;
	
	mat3 normalMtx = mat3(ViewMatrix * ModelMatrix);
	oNormal = normalMtx * normalize(iNormal);
	
	oTexCoord = iTexCoord;
}
