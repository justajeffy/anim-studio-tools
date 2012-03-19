#version 150 core

// FragmentShader Uniforms
uniform sampler2D TextureSampler0;

// Input from VertexShader
highp in vec3 oPosition;
highp in vec3 oNormal;
highp in vec2 oTexCoord;
highp in vec2 oOrgPosition;

// Final output color
out vec4 gl_FragData[ gl_MaxDrawBuffers ];

void main(void)
{
	vec4 tex0 = texture2D( TextureSampler0, oTexCoord );

	gl_FragData[0] = vec4( oPosition, oOrgPosition.x );
	
	gl_FragData[1] = vec4( normalize( oNormal), oOrgPosition.y );
	//gl_FragData[1] = vec4( normalize(cross(ddx(oPosition.xyz), ddy(oPosition.xyz))), 1);
	
	gl_FragData[2] = tex0;
}
