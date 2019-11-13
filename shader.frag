// Author:
// Title:

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

const int MAX_MARCHING_STEPS = 255;
const float MIN_DIST = 0.0;
const float MAX_DIST = 100.0;
const float EPSILON = 0.0001;

/**
 * Rotation matrix around the X axis.
 */
mat3 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(1, 0, 0),
        vec3(0, c, -s),
        vec3(0, s, c)
    );
}

/**
 * Rotation matrix around the Y axis.
 */
mat3 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, 0, s),
        vec3(0, 1, 0),
        vec3(-s, 0, c)
    );
}

/**
 * Constructive solid geometry union operation on SDF-calculated distances.
 */
float unionSDF(float distA, float distB) {
    return min(distA, distB);
}}

/**
 * translation matrix factory.
 */
mat4 translationMatrix(float x, float y, float z) {
    return mat4(
        vec4(1, 0, 0, x),
        vec4(0, 1, 0, y),
        vec4(0, 0, 1, z),
        vec4(0, 0, 0, 1)
    );
}

/**
 * translate 3D sample point.
 */
vec3 translatePoint(vec3 p, float x, float y, float z) {
    return (vec4(p, 1.0) * translationMatrix(-x, -y, -z)).xyz;
}

/**
 * Constructive solid geometry union operation on SDF-calculated distances.
 */
float unionSDF(float distA, float distB) {
    return min(distA, distB);
}

/**
 * Signed distance function for a round box centered at the origin
 * with dimensions specified by size.
 */
float roundBoxSDF(vec3 p, vec3 size, float r) {
    vec3 d = abs(p) - (size / 2.0);
    
    // Assuming p is inside the cube, how far is it from the surface?
    // Result will be negative or zero.
    float insideDistance = min(max(d.x, max(d.y, d.z)), 0.0) - r;
    
    // Assuming p is outside the cube, how far is it from the surface?
    // Result will be positive or zero.
    float outsideDistance = length(max(d, 0.0));
    
    return insideDistance + outsideDistance;
}

/**
 * Signed distance function describing the scene.
 * 
 * Absolute value of the return value indicates the distance to the surface.
 * Sign indicates whether the point is inside or outside the surface,
 * negative indicating inside.
 */
float fScene(vec3 samplePoint) {    
    // Slowly spin the whole scene
    samplePoint = rotateY(u_time / 2.0) * samplePoint;
    samplePoint = rotateX(u_time / 2.0) * samplePoint;

    
    float j0 = roundBoxSDF(translatePoint(samplePoint, -3.0, 0.0, 0.0), vec3(0.8, 0.8, 1.0), 0.1);
    float j1 = roundBoxSDF(translatePoint(samplePoint, -2.0, 0.5, 0.0), vec3(0.8, 1.8, 1.0), 0.1);
    float j2 = roundBoxSDF(translatePoint(samplePoint, -2.5, -0.2, 0.0), vec3(1.8, 0.4, 1.0), 0.1);
    float j = unionSDF(j2, unionSDF(j0, j1));

    float m0 = roundBoxSDF(translatePoint(samplePoint, -1.0, 0.5, 0.0), vec3(0.8, 1.8, 1.0), 0.1);
    float m1 = roundBoxSDF(translatePoint(samplePoint, 0.0, 1.0, 0.0), vec3(0.8, 0.8, 1.0), 0.1);
    float m2 = roundBoxSDF(translatePoint(samplePoint, 1.0, 0.5, 0.0), vec3(0.8, 1.8, 1.0), 0.1);
    float m3 = roundBoxSDF(translatePoint(samplePoint, 0.0, 1.2, 0.0), vec3(1.2, 0.4, 1.0), 0.1);
    float m = unionSDF(m3, unionSDF(m0, unionSDF(m1, m2)));

    float z0 = roundBoxSDF(translatePoint(samplePoint, 2.5, 1.0, 0.0), vec3(1.8, 0.8, 1.0), 0.1);
    float z1 = roundBoxSDF(translatePoint(samplePoint, 2.5, 0.0, 0.0), vec3(1.8, 0.8, 1.0), 0.1);
	float z2 = roundBoxSDF(translatePoint(samplePoint, 2.1, 0.5, 0.0), vec3(0.4, 0.8, 1.0), 0.1);
    float z = unionSDF(z2, unionSDF(z0, z1));
    
    
    return unionSDF(j, unionSDF(m, z));
}

/**
 * Return the shortest distance from the eyepoint to the scene surface along
 * the marching direction. If no part of the surface is found between start and end,
 * return end.
 * 
 * eye: the eye point, acting as the origin of the ray
 * marchingDirection: the normalized direction to march in
 * start: the starting distance away from the eye
 * end: the max distance away from the ey to march before giving up
 */
float shortestDistanceToSurface(vec3 eye, vec3 marchingDirection, float start, float end) {
    float depth = start;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = fScene(eye + depth * marchingDirection);
        if (dist < EPSILON) {
			return depth;
        }
        depth += dist;
        if (depth >= end) {
            return end;
        }
    }
    return end;
}

/**
 * Return the normalized direction to march in from the eye point for a single pixel.
 * 
 * fieldOfView: vertical field of view in degrees
 * size: resolution of the output image
 * fragCoord: the x,y coordinate of the pixel in the output image
 */
vec3 rayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - size / 2.0;
    // float z = size.y / tan(radians(fieldOfView) / 2.0);
    float z = size.y * 0.5 / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(xy, -z));
}

/**
 * Using the gradient of the SDF, estimate the normal on the surface at point p.
 */
vec3 estimateNormal(vec3 p) {
    return normalize(vec3(
        fScene(vec3(p.x + EPSILON, p.y, p.z)) - fScene(vec3(p.x - EPSILON, p.y, p.z)),
        fScene(vec3(p.x, p.y + EPSILON, p.z)) - fScene(vec3(p.x, p.y - EPSILON, p.z)),
        fScene(vec3(p.x, p.y, p.z  + EPSILON)) - fScene(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

/**
 * Lighting contribution of a single point light source via Phong illumination.
 * 
 * The vec3 returned is the RGB color of the light's contribution.
 *
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 * lightPos: the position of the light
 * lightIntensity: color/intensity of the light
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
vec3 phongContribForLight(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye,
                          vec3 lightPos, vec3 lightIntensity) {
    vec3 N = estimateNormal(p);
    vec3 L = normalize(lightPos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));
    
    float dotLN = clamp(dot(L, N), 0.0, 1.0);
    float dotRV = dot(R, V);
    
    if (dotLN < 0.0) {
        // Light not visible from this point on the surface
        return vec3(0.0, 0.0, 0.0);
    } 
    
    if (dotRV < 0.0) {
        // Light reflection in opposite direction as viewer, apply only diffuse
        // component
        return lightIntensity * (k_d * dotLN);
    }
    return lightIntensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
}

/**
 * Lighting via Phong illumination.
 * 
 * The vec3 returned is the RGB color of that point after lighting is applied.
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
vec3 phongIllumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye) {
    const vec3 ambientLight = 0.5 * vec3(1.0, 1.0, 1.0);
    vec3 color = ambientLight * k_a;
    
    vec3 light1Pos = vec3(4.0,
                          2.0,
                          4.0 * cos(u_time));
    vec3 light1Intensity = vec3(0.4, 0.4, 0.4);
    
    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  light1Pos,
                                  light1Intensity);
    
    vec3 light2Pos = vec3(2.0 * sin(0.37 * u_time),
                          2.0 * cos(0.37 * u_time),
                          2.0);
    vec3 light2Intensity = vec3(0.4, 0.4, 0.4);
    
    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  light2Pos,
                                  light2Intensity);    
    return color;
}

/**
 * Return a transform matrix that will transform a ray from view space
 * to world coordinates, given the eye point, the camera target, and an up vector.
 *
 * This assumes that the center of the camera is aligned with the negative z axis in
 * view space when calculating the ray marching direction. See rayDirection.
 */
mat4 viewMatrix(vec3 eye, vec3 center, vec3 up) {
    // Based on gluLookAt man page
    vec3 f = normalize(center - eye);
    vec3 s = normalize(cross(f, up));
    vec3 u = normalize(cross(s, f));
    return mat4(
        vec4(s, 0.0),
        vec4(u, 0.0),
        vec4(-f, 0.0),
        vec4(0.0, 0.0, 0.0, 1)
    );
}

void main() {
	vec3 viewDir = rayDirection(45.0, u_resolution, gl_FragCoord.xy);
    vec3 eye = vec3(8.0, 5.0, 7.0);
    
    mat4 viewToWorld = viewMatrix(eye, vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));
    
    vec3 worldDir = (viewToWorld * vec4(viewDir, 0.0)).xyz;
    
    float dist = shortestDistanceToSurface(eye, worldDir, MIN_DIST, MAX_DIST);
    
    if (dist > MAX_DIST - EPSILON) {
        // Didn't hit anything
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
		return;
    }
    
    // The closest point on the surface to the eyepoint along the view ray
    vec3 p = eye + dist * worldDir;
    
    vec3 K_a = vec3(0.2, 0.2, 0.2);
    vec3 K_d = vec3(0.7, 0.2, 0.2);
    vec3 K_s = vec3(1.0, 1.0, 1.0);
    float shininess = 10.0;
    
    vec3 color = phongIllumination(K_a, K_d, K_s, shininess, p, eye);
    
    gl_FragColor = vec4(color, 1.0);
}
