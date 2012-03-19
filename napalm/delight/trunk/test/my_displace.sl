displacement my_displace(
    float Km = 1.0;
    float amplitude = 1.0;
    string texturename = "";
    float __displacementbound_sphere = Km * amplitude;
    string __displacementbound_coordinatesystem = "current" )
{
    if( texturename != "" )
    {
        float amp = Km * amplitude * float texture( texturename, s, t );

        P += amp * normalize(N);
        N = calculatenormal( P );
    }
}
