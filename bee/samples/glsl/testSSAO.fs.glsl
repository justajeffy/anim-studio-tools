#version 150 core

// FragmentShader Uniforms
uniform sampler2D TextureSampler0; // viewPos
uniform sampler2D TextureSampler1; // viewNormal
uniform vec4 TextureSamplerSize1;
uniform sampler2D TextureSampler2; // albedo
uniform sampler2D TextureSampler3; // noise

uniform vec4 Light0PositionAttenuation;
uniform vec3 Light0Color;
uniform mat4 ViewMatrix;

// Input from VertexShader
highp in vec4 oTexCoord;
highp in vec4 oColor;

// Final output color
highp out vec4 outColor;



uniform vec2 g_Resolution;
uniform vec2 g_InvResolution;
uniform float g_R;
uniform vec2 g_FocalLen;
uniform vec2 g_InvFocalLen;
uniform float g_NumSteps;
uniform float g_NumDir;
uniform float g_AngleBias;
uniform float g_Attenuation;
uniform float g_inv_R;
uniform float g_sqr_R;

uniform float g_cameraRange;

//////////////////////////////////////////////////////////////////////////////

#define M_PI 3.14159265f


float length2(vec3 v) { return dot(v, v); }

vec3 min_diff(vec3 P, vec3 Pr, vec3 Pl)
{
    vec3 V1 = Pr - P;
    vec3 V2 = P - Pl;
    return (length2(V1) < length2(V2)) ? V1 : V2;
}

vec3 uv_to_eye(vec2 uv, float eye_z)
{
    uv = (uv * vec2(2.0, -2.0) - vec2(1.0, -1.0));
    return vec3(uv * g_InvFocalLen * eye_z, eye_z);
}

vec3 fetch_eye_pos(vec2 uv)
{
    //float z = texture2D(TextureSampler0, uv).z;
    //return uv_to_eye(uv, z);
    
    return texture2D(TextureSampler0, uv).rgb;

    //float z = texture2D(TextureSampler0, uv).b;
    //return uv_to_eye(uv, z);
}

float getEyeSpaceDepth(vec3 P)
{ // in 0,+1
	return length(P) / g_cameraRange;
}

vec3 tangent_eye_pos(vec2 uv, vec4 tangentPlane)
{
    // view vector going through the surface point at uv
    vec3 V = fetch_eye_pos(uv);
    float NdotV = dot(tangentPlane.xyz, V);
    // intersect with tangent plane except for silhouette edges
    if (NdotV < 0.0) V *= (tangentPlane.w / NdotV);
    return V;
}

float falloff(float r)
{
    return 1.0f - g_Attenuation*r*r;
}

vec2 snap_uv_offset(vec2 uv)
{
    return round(uv * g_Resolution) * g_InvResolution;
}

vec2 snap_uv_coord(vec2 uv)
{
    //return (floor(uv * g_Resolution) + 0.5f) * g_InvResolution;
    return uv - (frac(uv * g_Resolution) - 0.5f) * g_InvResolution;
}

float tan_to_sin(float x)
{
    return x / sqrt(1.0f + x*x);
}

vec3 tangent_vector(vec2 deltaUV, vec3 dPdu, vec3 dPdv)
{
    return deltaUV.x * dPdu + deltaUV.y * dPdv;
}

float tangent(vec3 T)
{
    return -T.z / length(T.xy);
}

float tangent(vec3 P, vec3 S)
{
    return (P.z - S.z) / length(S.xy - P.xy);
}

float biased_tangent(vec3 T)
{
    float phi = atan(tangent(T)) + g_AngleBias;
    return tan(min(phi, M_PI*0.5));
}

vec2 rotate_direction(vec2 Dir, vec2 CosSin)
{
    return vec2(Dir.x*CosSin.x - Dir.y*CosSin.y,
                Dir.x*CosSin.y + Dir.y*CosSin.x);
}

void integrate_direction(inout float ao, vec3 P, vec2 uv, vec2 deltaUV,
                         float numSteps, float tanH, float sinH)
{
    for (float j = 1; j <= numSteps; ++j)
    {
        uv += deltaUV;
        vec3 S = fetch_eye_pos(uv);

        // Ignore any samples outside the radius of influence
        float d2  = length2(S - P);
        if (d2 < g_sqr_R)
        {
            float tanS = tangent(P, S);

            if(tanS > tanH)
            {
                // Accumulate AO between the horizon and the sample
                float sinS = tanS / sqrt(1.0f + tanS*tanS);
                float r = sqrt(d2) * g_inv_R;
                ao += falloff(r) * (sinS - sinH);

                // Update the current horizon angle
                tanH = tanS;
                sinH = sinS;
            }
        }
    }
}

float AccumulatedHorizonOcclusion_LowQuality(vec2 deltaUV,
                                             vec2 uv0,
                                             vec3 P,
                                             float numSteps,
                                             float randstep)
{
    // Randomize starting point within the first sample distance
    vec2 uv = uv0 + snap_uv_offset( randstep * deltaUV );

    // Snap increments to pixels to avoid disparities between xy
    // and z sample locations and sample along a line
    deltaUV = snap_uv_offset( deltaUV );

    float tanT = tan(-M_PI*0.5 + g_AngleBias);
    float sinT = (g_AngleBias != 0.0) ? tan_to_sin(tanT) : -1.0;

    float ao = 0;
    integrate_direction(ao, P, uv, deltaUV, numSteps, tanT, sinT);

    // Integrate opposite directions together
    deltaUV = -deltaUV;
    uv = uv0 + snap_uv_offset( randstep * deltaUV );
    integrate_direction(ao, P, uv, deltaUV, numSteps, tanT, sinT);

    // Divide by 2 because we have integrated 2 directions together
    // Subtract 1 and clamp to remove the part below the surface
    return max(ao * 0.5 - 1.0, 0.0);
}

float AccumulatedHorizonOcclusion(vec2 deltaUV,
                                  vec2 uv0,
                                  vec3 P,
                                  float numSteps,
                                  float randstep,
                                  vec3 dPdu,
                                  vec3 dPdv )
{
    // Randomize starting point within the first sample distance
    vec2 uv = uv0 + snap_uv_offset( randstep * deltaUV );

    // Snap increments to pixels to avoid disparities between xy
    // and z sample locations and sample along a line
    deltaUV = snap_uv_offset( deltaUV );

    // Compute tangent vector using the tangent plane
    vec3 T = deltaUV.x * dPdu + deltaUV.y * dPdv;

    float tanH = biased_tangent(T);
    float sinH = tanH / sqrt(1.0f + tanH*tanH);

    float ao = 0;
    for(float j = 1; j <= numSteps; ++j)
    {
        uv += deltaUV;
        vec3 S = fetch_eye_pos(uv);

        // Ignore any samples outside the radius of influence
        float d2  = length2(S - P);
        if (d2 < g_sqr_R)
        {
            float tanS = tangent(P, S);

            if(tanS > tanH)
            {
                // Accumulate AO between the horizon and the sample
                float sinS = tanS / sqrt(1.0f + tanS*tanS);
                float r = sqrt(d2) * g_inv_R;
                ao += falloff(r) * (sinS - sinH);

                // Update the current horizon angle
                tanH = tanS;
                sinH = sinS;
            }
        }
    }

    return ao;
}

void main(void)
{
	vec3 viewPosition = texture2D(TextureSampler0, oTexCoord.xy).rgb;
	vec3 viewNormal = texture2D(TextureSampler1, oTexCoord.xy).rgb;
	vec3 albedo = texture2D(TextureSampler2, oTexCoord.xy).rgb;
	vec3 norViewNormal = normalize(viewNormal);
	
	vec2 orgPosXZ = vec2( texture2D(TextureSampler0, oTexCoord.xy).w, texture2D(TextureSampler1, oTexCoord.xy).w );

	vec3 emissive = vec3(0,0,0);

	vec3 LightDir = Light0PositionAttenuation.xyz - viewPosition;
	float lightDist = length( LightDir );
	LightDir = normalize( LightDir );
	float lightAtt = 1 - lightDist * Light0PositionAttenuation.w;

	outColor.rgb = emissive;
	outColor.rgb += vec3( max(0, dot(norViewNormal, LightDir)) ) * lightAtt * Light0Color;
	outColor.rgb *= albedo;

	outColor.rgb = albedo;
	outColor.a = 1;

	// ssao stuff

   vec3 P = fetch_eye_pos(oTexCoord.xy);
	vec3 N = norViewNormal;

	float esDepth = getEyeSpaceDepth(P);

	// setposition from -1,+1 to 0,+1 :
	//P = P * 0.5 + 0.5;

	// Project the radius of influence g_R from eye space to texture space.
	// The scaling by 0.5 is to go from [-1,1] to [0,1].
	//vec2 step_size = 0.5 * g_R  * g_FocalLen / esDepth;
	vec2 step_size = g_R  * g_FocalLen / esDepth;

	// Early out if the projected radius is smaller than 1 pixel.
	float numSteps = min ( g_NumSteps, min(step_size.x * g_Resolution.x, step_size.y * g_Resolution.y));

	float ao = 0;
	if( numSteps < 1.0 )
	{
		ao = 1;

		outColor = vec4(1,0,0,1);
	}
	else
	{
		step_size = step_size / ( numSteps + 1 );

		// Nearest neighbor pixels on the tangent plane
		vec3 Pr, Pl, Pt, Pb;
	   vec4 tangentPlane;

		tangentPlane = vec4(N, dot(P, N));
		Pr = tangent_eye_pos(oTexCoord.xy + vec2(g_InvResolution.x, 0), tangentPlane);
		Pl = tangent_eye_pos(oTexCoord.xy + vec2(-g_InvResolution.x, 0), tangentPlane);
		Pt = tangent_eye_pos(oTexCoord.xy + vec2(0, g_InvResolution.y), tangentPlane);
		Pb = tangent_eye_pos(oTexCoord.xy + vec2(0, -g_InvResolution.y), tangentPlane);

		// Screen-aligned basis for the tangent plane
		vec3 dPdu = min_diff(P, Pr, Pl);
		vec3 dPdv = min_diff(P, Pt, Pb) * (g_Resolution.y * g_InvResolution.x);

		// (cos(alpha),sin(alpha),jitter)
		//vec3 rand = tRandom.Load(int3((int)IN.pos.x&63, (int)IN.pos.y&63, 0)).xyz;
		//vec3 rand = texture2D(TextureSampler3, oTexCoord.zw).rgb;
		//vec3 rand = texture2D(TextureSampler3, viewPosition.xy).rgb;
		vec3 rand = texture2D(TextureSampler3, orgPosXZ).rgb;		

		float d;
		float alpha = 2.0f * M_PI / g_NumDir;
		
		float L = length(P);
		float cumul = 0;

		for (d = 0; d < g_NumDir*0.5; ++d)
		{
			float angle = alpha * d;
			vec2 dir = vec2(cos(angle), sin(angle));
			vec2 deltaUV = 2*rotate_direction(dir, rand.xy) * step_size.xy - 1;
			//ao += AccumulatedHorizonOcclusion(deltaUV, oTexCoord.xy, P, numSteps, rand.z, dPdu, dPdv);
			
			vec3 nP = fetch_eye_pos(oTexCoord.xy + deltaUV);
			float nL = length(nP); 
			
			float diff = abs( nL - L );
			cumul += diff;
		}

		//ao *= 2.0;
		ao = cumul / (g_NumDir*0.5);

		outColor = (1 - ao).xxxx;
		//outColor.rgb = rand.zzz;

		//outColor.rgb = dPdu;
	}



	//outColor = OUTPUT;
}

