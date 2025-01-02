import numpy as np
import os
def standardize_bbox(pcl, dis, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices] # n by 3
    dis_pt=dis[pt_indices]
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    return result, dis_pt

xml_head = \
"""
<scene version="0.5.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="ldrfilm">
            <integer name="width" value="1600"/>
            <integer name="height" value="1200"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

xml_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="0.008"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""

def colormap(x,y,z):
    vec = np.array([x,y,z])
    vec = np.clip(vec, 0.001,1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]

def error_color(dis):
    r=dis
    g=dis
    b=1-dis
    return [r, g, b]
def get_xml(pcl,out_file):
    xml_segments = [xml_head]

    pcl,dis = standardize_bbox(pcl[:,:3],pcl[:,-1], pcl.shape[0])
    pcl = pcl[:, [2, 0, 1]]
    # pcl[:, 0] *= -1
    pcl[:, 2] += -0.0125

    for i in range(pcl.shape[0]):
        # color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
        color=error_color(dis[i])
        xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)

    with open(out_file, 'w') as f:
        f.write(xml_content)
def rot(points,alpha,beta,theta):
    alpha=alpha/180*np.pi
    beta=beta/180*np.pi
    theta=theta/180*np.pi
    X=[[1,0,0],
       [0,np.cos(alpha),-np.sin(alpha)],
       [0,np.sin(alpha),np.cos(alpha)]]
    Y=[[np.cos(beta),0,np.sin(beta)],
       [0,1,0],
       [-np.sin(beta),0,np.cos(beta)]]
    Z=[[np.cos(theta),-np.sin(theta),0],
       [np.sin(theta),np.cos(theta),0],
       [0,0,1]]
    M=np.matmul(X,Y)
    M=np.matmul(M,Z)
    points=np.matmul(points,M)
    return points


def main():
    dir=''
    save_dir=''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for file in os.listdir(dir):
        data=np.loadtxt(dir+file)
        data[:,:3]=rot(data[:,:3],90,45,0)
        out_file=save_dir+file[:-3]+'xml'
        get_xml(data,out_file)
if __name__ == '__main__':
    main()