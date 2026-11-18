try:
    from . import generic as g
except BaseException:
    import generic as g

from collections import Counter


def test_rabbit():
    # A BSD-licensed test model from pyglet
    # it has mixed triangles, quads, and 16 element faces -_-
    # this should test the non-vectorized load path
    m = g.get_mesh("rabbit.obj")
    assert len(m.faces) == 1252
    rec = g.roundtrip(m.export(file_type="obj"), file_type="obj")
    assert g.np.isclose(m.area, rec.area)


def test_no_img():
    # sometimes people use the `vt` parameter for arbitrary
    # vertex attributes and thus want UV coordinates even
    # if there is no texture image
    m = g.get_mesh("noimg.obj")
    assert m.visual.uv.shape == (len(m.vertices), 2)
    # make sure UV coordinates are in range 0.0 - 1.0
    assert m.visual.uv.max() < (1 + 1e-5)
    assert m.visual.uv.min() > -1e-5
    # check to make sure it's not all zeros
    assert g.np.ptp(m.visual.uv) > 0.5
    rec = g.roundtrip(m.export(file_type="obj"), file_type="obj")
    assert g.np.isclose(m.area, rec.area)


def test_keep_unreferenced():
    # try loading a short mesh with 2 vertices, one of which is referenced
    m = g.trimesh.load(
        g.trimesh.util.wrap_as_stream("o mesh\nv 1 1 1\nv 1 1 2\nf 1 1 1"),
        file_type="obj",
        process=False,
        maintain_order=True,
    )

    assert g.np.allclose(m.faces[0], [0, 0, 0])
    assert g.np.allclose(m.vertices, [[1, 1, 1], [1, 1, 2]])


def test_trailing():
    # test files with texture and trailing slashes
    m = g.get_mesh("jacked.obj")
    assert len(m.visual.uv) == len(m.vertices)
    rec = g.roundtrip(m.export(file_type="obj"), file_type="obj")
    assert g.np.isclose(m.area, rec.area)


def test_obj_groups():
    # a wavefront file with groups defined
    mesh = g.get_mesh("groups.obj")

    # make sure some data got loaded
    assert g.trimesh.util.is_shape(mesh.faces, (-1, 3))
    assert g.trimesh.util.is_shape(mesh.vertices, (-1, 3))

    # make sure groups are the right length
    # TODO: we do not support face groups now
    # assert len(mesh.metadata['face_groups']) == len(mesh.faces)

    # check to make sure there is signal not just zeros
    # assert g.np.ptp(mesh.metadata['face_groups']) > 0


def test_obj_groups_split():
    # a wavefront file with 14 groups and 2 objects defined
    mesh = g.get_mesh("groups.obj", split_groups=True)
    # make sure the right number of meshes are split
    assert len(mesh.geometry) == 14

    # keys of (split_object, split_groups) arguments
    # values of how many meshes should be output for `groups.obj`
    checks = {(True, True): 15, (True, False): 2, (False, True): 14, (False, False): 1}

    for (split_objects, split_groups), truth in checks.items():
        scene = g.get_scene(
            "groups.obj", split_objects=split_objects, split_groups=split_groups
        )
        # make sure we produced the expected number of geometries
        assert len(scene.geometry) == truth


def test_obj_negative_indices():
    # a wavefront file with negative indices
    mesh = g.get_mesh("negative_indices.obj", merge_tex=True, merge_norm=True)

    # make sure some data got loaded
    assert mesh.faces.shape == (12, 3)
    assert mesh.vertices.shape == (8, 3)


def test_obj_quad():
    mesh = g.get_mesh("quadknot.obj", merge_tex=True)
    # make sure some data got loaded
    assert g.trimesh.util.is_shape(mesh.faces, (-1, 3))
    assert g.trimesh.util.is_shape(mesh.vertices, (-1, 3))

    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    rec = g.roundtrip(mesh.export(file_type="obj"), file_type="obj")
    assert g.np.isclose(mesh.area, rec.area)


def test_obj_multiobj():
    # test a wavefront file with multiple objects in the same file
    scene = g.get_mesh("two_objects.obj", split_objects=True, group_material=False)
    assert len(scene.geometry) == 2

    for mesh in scene.geometry.values():
        # make sure some data got loaded
        assert g.trimesh.util.is_shape(mesh.faces, (-1, 3))
        assert g.trimesh.util.is_shape(mesh.vertices, (-1, 3))

        assert mesh.is_watertight
        assert mesh.is_winding_consistent


def test_obj_split_attributes():
    # test a wavefront file where pos/uv/norm have different indices
    # and where multiple objects share vertices
    # Note 'process=False' to avoid merging vertices
    scene = g.get_mesh(
        "joined_tetrahedra.obj",
        process=False,
        split_objects=True,
        group_material=False,
    )

    assert len(scene.geometry) == 2

    geom = list(scene.geometry.values())

    assert g.trimesh.util.is_shape(geom[0].faces, (4, 3))
    assert g.trimesh.util.is_shape(geom[0].vertices, (9, 3))
    assert g.trimesh.util.is_shape(geom[1].faces, (4, 3))
    assert g.trimesh.util.is_shape(geom[1].vertices, (9, 3))


def test_obj_simple_order():
    # test a simple wavefront model without split indexes
    # and make sure we don't reorder vertices unnecessarily
    file_name = g.os.path.join(g.dir_models, "cube.OBJ")
    # load a simple OBJ file without merging vertices
    m = g.trimesh.load(file_name, process=False)
    # use trivial loading to compare with fancy performant one
    with open(file_name) as f:
        f, v, _vt = simple_load(f.read())
    # trimesh loader should return the same face order
    assert g.np.allclose(f, m.faces)
    assert g.np.allclose(v, m.vertices)


def test_order_tex():
    # test a simple wavefront model without split indexes
    # and make sure we don't reorder vertices unnecessarily
    file_name = g.os.path.join(g.dir_models, "fuze.obj")
    # load a simple OBJ file without merging vertices
    m = g.trimesh.load(file_name, process=False, maintain_order=True)
    # use trivial loading to compare with fancy performant one
    with open(file_name) as f:
        f, v, _vt = simple_load(f.read())
    # trimesh loader should return the same face order
    assert g.np.allclose(f, m.faces)
    assert g.np.allclose(v, m.vertices)


def test_obj_compressed():
    mesh = g.get_mesh("cube_compressed.obj", process=False)
    assert mesh._cache.cache["vertex_normals"].shape == mesh.vertices.shape
    assert g.np.allclose(g.np.abs(mesh.vertex_normals).sum(axis=1), 1.0)


def test_vertex_color():
    # get a box mesh
    mesh = g.trimesh.creation.box()
    # set each vertex to a unique random color
    mesh.visual.vertex_colors = [
        g.trimesh.visual.random_color() for _ in range(len(mesh.vertices))
    ]
    # export and then reload the file as OBJ
    rec = g.trimesh.load(
        g.trimesh.util.wrap_as_stream(mesh.export(file_type="obj")), file_type="obj"
    )

    # assert colors have survived the export cycle
    assert (mesh.visual.vertex_colors == rec.visual.vertex_colors).all()


def test_single_vn():
    """
    Make sure files with a single VN load.
    """
    m = g.get_mesh("singlevn.obj")
    assert len(m.vertices) > 0
    assert len(m.faces) > 0


def test_polygon_faces():
    m = g.get_mesh("polygonfaces.obj")
    assert len(m.vertices) > 0
    assert len(m.faces) > 0


def test_faces_not_enough_indices():
    m = g.get_mesh("notenoughindices.obj")
    assert len(m.vertices) > 0
    assert len(m.faces) == 1


def test_export_path():
    m = g.get_mesh("fuze.obj")
    g.check_fuze(m)
    assert m._cache.cache["vertex_normals"].shape == m.vertices.shape
    with g.TemporaryDirectory() as d:
        file_path = g.os.path.join(d, "fz.obj")
        m.export(file_path)
        r = g.trimesh.load(file_path)
        g.check_fuze(r)


def test_skip_mtl():
    # not loading materials should produce a trivial texture
    m_tex = g.get_mesh("fuze.obj")
    m_tex_size = m_tex.visual.material.image.size

    m_notex = g.get_mesh("fuze.obj", skip_materials=True)
    m_notex_size = m_notex.visual.material.image.size

    assert m_tex_size != m_notex_size


def test_mtl():
    # get a mesh with texture
    m = g.get_mesh("fuze.obj")
    # export the mesh including data
    obj, data = g.trimesh.exchange.export.export_obj(m, return_texture=True)
    with g.TemporaryDirectory() as path:
        # where is the OBJ file going to be saved
        obj_path = g.os.path.join(path, "test.obj")
        with open(obj_path, "w") as f:
            f.write(obj)
        # save the MTL and images
        for k, v in data.items():
            with open(g.os.path.join(path, k), "wb") as f:
                f.write(v)
        # reload the mesh from the export
        rec = g.trimesh.load(obj_path)
    # make sure loaded image is the same size as the original
    assert rec.visual.material.image.size == m.visual.material.image.size
    # make sure the faces are the same size
    assert rec.faces.shape == m.faces.shape


def test_scene():
    s = g.get_mesh("cycloidal.3DXML")

    e = g.trimesh.load(
        g.io_wrap(s.export(file_type="obj")),
        file_type="obj",
        split_object=True,
        group_materials=False,
    )

    assert g.np.isclose(e.area, s.area, rtol=0.01)


def test_edge_cases():
    # a mesh with some NaN colors
    n = g.get_mesh("nancolor.obj")
    assert n.faces.shape == (12, 3)

    v = g.get_mesh("cubevt.obj")
    assert v.faces.shape == (12, 3)


def test_empty_or_pointcloud():
    # demo files to check
    empty_files = ["obj_empty.obj", "obj_points.obj", "obj_wireframe.obj"]

    for empty_file in empty_files:
        e = g.get_mesh("emptyIO/" + empty_file)

        # create export
        if "empty" in empty_file:
            try:
                export = e.export(file_type="ply")
            except BaseException:
                continue
            raise ValueError("cannot export empty")
        elif "points" in empty_file:
            export = e.export(file_type="ply")
            reconstructed = g.roundtrip(export, file_type="ply")

            # result should be a point cloud instance
            assert isinstance(e, g.trimesh.PointCloud)
            assert hasattr(e, "vertices")
            # point cloud export should contain vertices
            assert isinstance(reconstructed, g.trimesh.PointCloud)
            assert hasattr(reconstructed, "vertices")


def test_backslash_continuation_character():
    # an obj file with \ (backslash) line continuation characters
    m = g.get_mesh("wallhole.obj")
    assert m.faces.shape == (66, 3)


def test_no_uv():
    mesh = g.get_mesh("box.obj")
    rec = g.roundtrip(mesh.export(file_type="obj"), file_type="obj")
    assert g.np.isclose(mesh.area, rec.area)


def test_no_uv_but_mtl():
    sphere = g.trimesh.creation.uv_sphere()
    sphere.visual = g.trimesh.visual.TextureVisuals(
        uv=None, material=g.trimesh.visual.material.empty_material()
    )
    output = sphere.export("sphere.obj")
    assert "usemtl" in output


def test_chair():
    mesh = next(iter(g.get_mesh("chair.zip").geometry.values()))

    # this model comes with vertex normals
    assert "vertex_normals" in mesh._cache
    assert g.np.allclose(1.0, g.np.linalg.norm(mesh.vertex_normals, axis=1))
    mesh.apply_scale(0.46377314288075433)
    assert "vertex_normals" in mesh._cache
    assert g.np.allclose(1.0, g.np.linalg.norm(mesh.vertex_normals, axis=1))
    assert "vertex_normals" in mesh._cache
    mesh._cache.clear()
    assert "vertex_normals" not in mesh._cache
    # if we recomputed now, the degenerate faces
    # would lead some of these vertex normals to be zero
    # assert g.np.allclose(
    #    1.0, g.np.linalg.norm(mesh.vertex_normals, axis=1))


def test_multi_nodupe():
    s = g.get_mesh("forearm.zae")
    obj, mtl = g.trimesh.exchange.obj.export_obj(
        s, include_color=True, include_texture=True, return_texture=True
    )
    # should be using one material file
    assert obj.count("mtllib") == 1
    assert "mtllib material.mtl" in obj
    # should be specifying 5 materials
    assert obj.count("usemtl") == 5

    # this file has only the properties (no images)
    assert len(mtl) == 1
    mtl_names = [
        L.strip().split()[-1].strip()
        for L in mtl["material.mtl"].decode("utf-8").split("\n")
        if "newmtl" in L
    ]

    # there should be 5 unique material names
    assert len(set(mtl_names)) == 5


def test_mtl_color_roundtrip():
    # create a mesh with a simple material
    m = g.trimesh.creation.box()
    m.visual = m.visual.to_texture()
    # set each color component to a unique value
    colors = [g.trimesh.visual.color.random_color() for _ in range(3)]
    m.visual.material.ambient = colors[0]
    m.visual.material.specular = colors[1]
    m.visual.material.diffuse = colors[2]
    m.visual.material.glossiness = 0.52622

    with g.TemporaryDirectory() as d:
        # exporting by filename will automatically
        # create a FilePathResolver which writes the
        # `mtl` file to the same directory
        file_name = g.os.path.join(d, "hi.obj")
        m.export(file_name)
        # reload the export by file name
        r = g.trimesh.load(file_name)

    # these values should have survived the roundtrip
    assert g.np.allclose(m.visual.material.ambient, r.visual.material.ambient)
    assert g.np.allclose(m.visual.material.specular, r.visual.material.specular)
    assert g.np.allclose(m.visual.material.diffuse, r.visual.material.diffuse)
    assert g.np.isclose(m.visual.material.glossiness, r.visual.material.glossiness)


def test_scene_export_material_name():
    s = g.get_mesh("fuze.obj", force="scene")

    g.log.warning(s.geometry)

    dummy = "fuxx"
    s.geometry["fuze.obj"].visual.material.name = dummy

    r = g.trimesh.resolvers.ZipResolver()
    r["model.obj"] = s.export(file_type="obj", mtl_name="mystuff.mtl", resolver=r)

    mtl = r["mystuff.mtl"].decode("utf-8")
    assert mtl.count("newmtl") == 1
    assert f"newmtl {dummy}" in mtl
    assert f"{dummy}.jpeg" in r


def test_compound_scene_export():
    # generate a mesh with multiple textures
    a = g.get_mesh("BoxTextured.glb")
    a = a.scaled(1.0 / a.extents.max())
    a.apply_translation(-a.bounds[0])

    b = g.get_mesh("fuze.obj").scene()
    b = b.scaled(1.0 / b.extents.max())
    b.apply_translation(-b.bounds[0] + [2, 0, 0])

    d = next(iter(b.copy().geometry.values()))
    d.apply_translation([-1, 0, 0])
    assert hash(d.visual.material) == hash(b.geometry["fuze.obj"].visual.material)

    # should change the material hash
    d.visual.material.glossiness = 0.1
    assert hash(d.visual.material) != hash(b.geometry["fuze.obj"].visual.material)

    # generate a compound scene
    c = a + b + d
    for i in c.geometry.values():
        # name all the materials the same thing
        i.visual.material.name = "material_0"

    # export the compound scene
    obj, mtl = c.export(file_type="obj", return_texture=True)
    # there should be exactly one mtllib referenced
    assert obj.count("mtllib") == 1
    assert obj.count("usemtl") == 3

    # should be one texture image for each of 3
    # plus the `.mtl` file itself
    # if we had image-hash-deduplication this should
    # be changed to 3 as the image for `b` and `d` are the same
    assert len(mtl) == 4

    # get the material names specified

    mtl_names = [
        L.strip().split()[-1].strip()
        for L in mtl["material.mtl"].decode("utf-8").split("\n")
        if "newmtl" in L
    ]
    # there should be 3 unique material names
    assert len(set(mtl_names)) == 3

    # now reload the compound scene
    t = g.trimesh.load(
        file_obj=g.trimesh.util.wrap_as_stream(obj),
        file_type="obj",
        resolver=g.trimesh.resolvers.ZipResolver(mtl),
        group_material=False,
        split_object=True,
    )
    # these names should match eventually
    assert len(t.geometry.keys()) == len(c.geometry.keys())
    assert g.np.isclose(t.area, c.area)


def test_face_parsing_in_group_names():
    # Checks that an obj with a g tag containinig a face like name (an 'f '
    # followed by three space separated text chunks, ex: f 1 2 3) does load
    # properly
    m = g.get_mesh("face_in_group_name.obj")
    assert len(m.vertices) == 1


def test_face_parsing_in_group_names_with_object_tag():
    # Checks that an obj with a g tag in the middle of a file,
    # containinig a face like name (an 'f ' followed by three
    # space separated text chunks, ex: f 1 2 3), does load properly
    m = g.get_mesh("face_in_group_name_mid_file.obj")
    assert len(m.vertices) == 5
    assert len(m.faces) == 2


def test_chunk_parsing_with_no_faces_but_with_f_in_chunk():
    # Checks that a chunk with no faces but with 'f ' in it loads properly
    m = g.get_mesh("obj_with_no_face_in_chunk.obj")
    assert len(m.vertices) == 3
    assert len(m.faces) == 1


def test_export_normals():
    m = g.trimesh.creation.box()
    assert "vertex_normals" not in m._cache.cache

    e = m.export(file_type="obj", include_normals=None)
    assert "vn " not in e
    e = m.export(file_type="obj", include_normals=True)
    # should have included normals
    assert "vn " in e
    # should have forced generation of normals
    assert "vertex_normals" in m._cache.cache

    # now that they're in cache include_normals=None should get them
    e = m.export(file_type="obj", include_normals=None)
    assert "vn " in e

    # or skip them if explicitly asked
    e = m.export(file_type="obj", include_normals=False)
    assert "vn " not in e


def test_export_mtl_args():
    mesh = g.trimesh.creation.box()
    # check for a crash with no materials defined
    _a, _b = g.trimesh.exchange.obj.export_obj(
        mesh, return_texture=True, mtl_name="hi.mtl"
    )


def simple_load(text):
    # we're going to load faces in a basic text way
    # and compare the order from this method to the
    # trimesh loader, to see if we get the same thing
    # note that trimesh's extremely convoluted string
    # wangling is wildly faster than this
    f = []
    v = []
    vt = []
    for line in str.splitlines(text):
        line = line.strip()
        if len(line) < 2:
            continue
        elif line.startswith("f "):
            if "/" in line:
                f.append([int(i.split("/", 1)[0]) for i in line[1:].strip().split()])
            else:
                f.append(line[1:].strip().split())
        elif line.startswith("v "):
            v.append(line[1:].strip().split())
        elif line.startswith("vt "):
            vt.append(line[2:].strip().split())

    # get faces as basic numpy array
    f = g.np.array(f, dtype=g.np.int64) - 1
    v = g.np.array(v, dtype=g.np.float64)
    vt = g.np.array(vt, dtype=g.np.float64)

    return f, v, vt


def test_material_name():
    m = g.get_mesh("fuze.obj")

    # check that the material used the name in the file
    assert m.visual.material.name == "a-super-duper-material"

    # roundtrip the OBJ through a file path
    # this is easier than making a resolver ourself for the MTL file
    # as it will automatically create the `FilePathResolver` on paths
    with g.TemporaryDirectory() as temp:
        path = g.os.path.join(temp, "mesh.obj")
        m.export(file_obj=path)
        roundtrip = g.trimesh.load_mesh(file_obj=path)

    # material name should have survived
    assert roundtrip.visual.material.name == "a-super-duper-material"

    g.check_fuze(roundtrip)


def test_obj_quad_uv():
    # check that the quads with UV's get translated
    m = g.get_mesh("quadknot.obj")

    assert m.visual.uv.shape[0] == len(m.vertices)
    assert g.np.isclose(m.visual.uv.max(), 0.1)


def test_no_vertices():
    import numpy as np

    import trimesh
    from trimesh.visual import ColorVisuals

    scene = trimesh.Scene()
    box = trimesh.creation.box(extents=[1, 2, 3])
    box.visual = ColorVisuals()
    box.visual.vertex_colors = np.random.uniform(size=(box.vertices.shape[0], 3))
    box.visual.face_colors = np.random.uniform(size=(box.faces.shape[0], 3))
    scene.add_geometry(box, geom_name="BOX")
    scene.add_geometry(trimesh.creation.cone(1.0, 1.0), geom_name="CONE")

    # roundtrip this test scene through an OBJ
    # box has 6-column `v` lines and the cone has 3-column `v` lines
    export = scene.export(file_type="obj")
    loaded = trimesh.load_scene(trimesh.util.wrap_as_stream(export), file_type="obj")

    # every vertex and face should have survived the mixed-width parse
    truth = scene.to_mesh()
    assert np.allclose(loaded.to_mesh().vertices, truth.vertices)
    assert np.array_equal(loaded.to_mesh().faces, truth.faces)

    # faces with no vertices should load as empty rather than `len(None)`
    empty = trimesh.load_scene(
        trimesh.util.wrap_as_stream("vn 0 0 1\nf 1 2 3\n"), file_type="obj"
    )
    assert len(empty.geometry) == 0


# ---------------------------------------------------------------------------
# round-trip semantics: shared positions do not imply shared normals / UVs
# ---------------------------------------------------------------------------


def _parse_obj_face_records(text):
    """
    A deliberately small and independent reference OBJ parser.

    Unlike the trimesh loader this keeps the three Wavefront index spaces
    (position `v`, texture `vt` and normal `vn`) fully separate, including
    negative relative indices, and returns one semantic record per face:
    each face is three corner tuples of

        (position, normal or None, uv or None, material name or None)

    Polygon faces are triangulated with the same first-vertex fan used by
    trimesh, so records can be compared across an export/import cycle even
    though the exporter only emits triangles. Two face *multisets* matching
    means every face still points at the same combination of position,
    normal, UV and material regardless of any legal vertex splitting, face
    reordering or cyclic corner permutation the loader/exporter performed.
    """
    v = []
    vt = []
    vn = []
    current_material = None
    faces = []

    def resolve(index, count):
        # wavefront indices are 1-based; negative indices are relative
        return index - 1 if index > 0 else count + index

    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        split = line.split()
        key = split[0]
        if key == "v":
            v.append([float(x) for x in split[1:4]])
        elif key == "vt":
            vt.append([float(x) for x in split[1:3]])
        elif key == "vn":
            vn.append([float(x) for x in split[1:4]])
        elif key == "usemtl":
            current_material = line[len("usemtl") :].strip()
        elif key == "f":
            refs = []
            for token in split[1:]:
                parts = token.split("/")
                iv = resolve(int(parts[0]), len(v))
                if len(parts) > 1 and parts[1]:
                    it = resolve(int(parts[1]), len(vt))
                else:
                    it = None
                if len(parts) > 2 and parts[2]:
                    inn = resolve(int(parts[2]), len(vn))
                else:
                    inn = None
                refs.append((iv, it, inn))
            # first-vertex fan triangulation matching trimesh
            for a, b, c in (
                (refs[0], refs[i], refs[i + 1]) for i in range(1, len(refs) - 1)
            ):
                faces.append(
                    tuple(
                        (
                            tuple(g.np.round(v[iv], 8)),
                            tuple(g.np.round(vn[inn], 8)) if inn is not None else None,
                            tuple(g.np.round(vt[it], 8)) if it is not None else None,
                            current_material,
                        )
                        for iv, it, inn in (a, b, c)
                    )
                )
    return faces


def _scene_face_records(scene):
    """
    Collect the same semantic per-face records from a loaded scene.

    Geometry is moved through its scene graph transform so positions are
    compared in world space, and the material assigned to each sub-mesh is
    attached to every one of its faces.
    """
    faces = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name]
        if not hasattr(mesh, "faces"):
            continue
        # the material each face of this geometry renders with
        material = getattr(getattr(mesh, "visual", None), "material", None)
        material_name = getattr(material, "name", None)
        # bake the instance transform into the positions
        points = g.trimesh.transform_points(mesh.vertices.copy(), transform)
        uv = getattr(mesh.visual, "uv", None)
        normals = mesh.vertex_normals if "vertex_normals" in mesh._cache.cache else None
        for face in mesh.faces:
            faces.append(
                tuple(
                    (
                        tuple(g.np.round(points[vertex_index], 8)),
                        tuple(g.np.round(normals[vertex_index], 8))
                        if normals is not None
                        else None,
                        tuple(g.np.round(uv[vertex_index], 8))
                        if uv is not None
                        else None,
                        material_name,
                    )
                    for vertex_index in face
                )
            )
    return faces


def _material_key(name):
    """
    Normalize a material name for semantic comparison.

    Meshes with UVs but no `usemtl` always render through an auto-generated
    empty material (``material_0``), so an explicit "no material" and the
    synthesized one are semantically identical. Explicitly named materials
    keep their names so cross-material stitching remains detectable.
    """
    if name is None:
        return None
    if str(name).startswith("material_") and str(name)[9:].isdigit():
        return None
    return name


def _canonical_face(face):
    """
    Canonicalize one face's three corners.

    A cyclic corner rotation keeps the same winding and is the same face,
    while a reversed order flips the orientation and must not compare
    equal. Material names are normalized through `_material_key`.
    """
    corners = tuple((p, n, t, _material_key(m)) for p, n, t, m in face)
    return min(corners[i:] + corners[:i] for i in range(3))


def _assert_face_records_match(a, b):
    """
    Compare per-face semantic records as multisets.

    The judgment is geometric/attribute based: a face is identified by the
    position+normal+UV+material tuples of its three corners, never by raw
    vertex indices or the total vertex count. Vertex splitting and
    reordering are therefore legal while cross-object or cross-material
    stitching changes the tuples and is detected here.
    """
    missing = Counter(map(_canonical_face, a)) - Counter(map(_canonical_face, b))
    extra = Counter(map(_canonical_face, b)) - Counter(map(_canonical_face, a))
    message = "\n".join(
        [
            "face semantics diverged across the OBJ round trip",
            f"missing records: {list(missing)[:3]}",
            f"unexpected records: {list(extra)[:3]}",
        ]
    )
    assert not missing and not extra, message


def test_obj_roundtrip_negative_indices_semantics():
    # the negative index model has separate position / UV / normal index
    # spaces including negative relative indices and quads: every face must
    # keep its position, normal and UV corner correspondence after export
    mesh = g.get_mesh(
        "negative_indices.obj", process=False, merge_tex=True, merge_norm=True
    )
    # sanity check on the source file itself
    with open(g.get_path("negative_indices.obj")) as file_obj:
        truth = _parse_obj_face_records(file_obj.read())
    assert len(truth) == len(mesh.faces)

    exported = mesh.export(file_type="obj")
    reloaded = g.trimesh.load_scene(
        g.trimesh.util.wrap_as_stream(exported),
        file_type="obj",
        process=False,
    )
    records = _scene_face_records(reloaded)

    # geometric/attribute semantics must survive, vertex count is only a
    # secondary check because unmerging is allowed to duplicate vertices
    _assert_face_records_match(truth, records)
    assert g.np.isclose(mesh.area, reloaded.area)


def _material_switch_obj():
    # two objects, two materials, shared positions across objects and
    # per-corner normals/UVs that differ even when positions coincide:
    # a shared position must not cause normals/UVs to be shared
    obj = """
mtllib switches.mtl
o alpha
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
vn 0 0 1
usemtl red
f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
o beta
v 0 0 0
v 0 0 1
v 1 0 1
v 1 0 0
vt 0.2 0.2
vt 0.8 0.2
vt 0.8 0.8
vt 0.2 0.8
vn 0 -1 0
usemtl blue
f 6/6/2 5/5/2 7/7/2
f 7/7/2 5/5/2 8/8/2
"""
    mtl = """
newmtl red
Kd 0.9 0.1 0.1
newmtl blue
Kd 0.1 0.1 0.9
"""
    return obj, mtl


def test_obj_roundtrip_multi_object_material_switch():
    obj, mtl = _material_switch_obj()
    truth = _parse_obj_face_records(obj)
    # the source really does switch both object and material
    assert {face[0][-1] for face in truth} == {"red", "blue"}

    resolver = g.trimesh.resolvers.ZipResolver({"switches.mtl": mtl.encode()})
    scene = g.trimesh.load_scene(
        g.trimesh.util.wrap_as_stream(obj),
        file_type="obj",
        resolver=resolver,
        process=False,
        split_objects=True,
        group_material=False,
    )
    assert len(scene.geometry) == 2

    exported, assets = g.trimesh.exchange.obj.export_obj(
        scene, return_texture=True, include_normals=True
    )
    reloaded = g.trimesh.load_scene(
        g.trimesh.util.wrap_as_stream(exported),
        file_type="obj",
        resolver=g.trimesh.resolvers.ZipResolver(assets),
        process=False,
        split_objects=True,
        group_material=False,
    )

    # every face keeps its position, normal, UV and material correspondence
    _assert_face_records_match(truth, _scene_face_records(reloaded))
    # auxiliary info only: splitting/indexing may change vertex totals
    assert reloaded.area == scene.area
    materials = {
        mesh.visual.material.name
        for mesh in reloaded.geometry.values()
        if getattr(mesh.visual, "material", None) is not None
    }
    assert materials == {"red", "blue"}


def test_obj_semantics_allow_vertex_split_and_reorder():
    # legitimate re-indexing: duplicate a shared corner and permute the
    # vertices, which must NOT be flagged as an error as long as every face
    # keeps the same position/normal/UV/material semantics
    obj, mtl = _material_switch_obj()
    truth = _parse_obj_face_records(obj)

    scene = g.trimesh.load_scene(
        g.trimesh.util.wrap_as_stream(obj),
        file_type="obj",
        resolver=g.trimesh.resolvers.ZipResolver({"switches.mtl": mtl.encode()}),
        process=False,
        split_objects=True,
        group_material=False,
    )
    mesh = next(iter(scene.geometry.values()))
    # append a fully consistent duplicate of the corner used by the first
    # face corner (same position, normal AND UV), then point that corner at
    # the duplicate and permute every row
    corner = mesh.faces[0, 0]
    vertices = g.np.vstack((mesh.vertices, mesh.vertices[corner]))
    normals = g.np.vstack((mesh.vertex_normals, mesh.vertex_normals[corner]))
    uv = g.np.vstack((mesh.visual.uv, mesh.visual.uv[corner]))
    faces = mesh.faces.copy()
    faces[0, 0] = len(vertices) - 1
    perm = g.np.random.default_rng(0).permutation(len(vertices))
    inverse = g.np.empty_like(perm)
    inverse[perm] = g.np.arange(len(perm))
    visual = g.trimesh.visual.TextureVisuals(uv=uv[perm], material=mesh.visual.material)
    reordered = g.trimesh.Trimesh(
        vertices=vertices[perm],
        faces=inverse[faces],
        vertex_normals=normals[perm],
        process=False,
        visual=visual,
    )
    exported, assets = g.trimesh.exchange.obj.export_obj(
        reordered, return_texture=True, include_normals=True
    )
    reloaded = g.trimesh.load_scene(
        g.trimesh.util.wrap_as_stream(exported),
        file_type="obj",
        resolver=g.trimesh.resolvers.ZipResolver(assets),
        process=False,
    )

    # same face semantics despite more vertices and a different ordering
    name = mesh.visual.material.name
    expected = [face for face in truth if face[0][-1] == name]
    _assert_face_records_match(expected, _scene_face_records(reloaded))


def test_obj_semantics_detect_cross_material_stitch():
    # corrupt a mesh by swapping UVs between the two materials so faces keep
    # their positions but render with the other material's texture coords:
    # a vertex-count based check would miss this, the semantic one does not
    obj, mtl = _material_switch_obj()
    scene = g.trimesh.load_scene(
        g.trimesh.util.wrap_as_stream(obj),
        file_type="obj",
        resolver=g.trimesh.resolvers.ZipResolver({"switches.mtl": mtl.encode()}),
        process=False,
        split_objects=True,
        group_material=False,
    )
    meshes = list(scene.geometry.values())
    uv_sets = [mesh.visual.uv.copy() for mesh in meshes]
    # stitch the wrong UV block onto each object
    meshes[0].visual.uv = uv_sets[1][: len(meshes[0].vertices)]
    meshes[1].visual.uv = uv_sets[0][: len(meshes[1].vertices)]

    exported, assets = g.trimesh.exchange.obj.export_obj(
        scene, return_texture=True, include_normals=True
    )
    reloaded = g.trimesh.load_scene(
        g.trimesh.util.wrap_as_stream(exported),
        file_type="obj",
        resolver=g.trimesh.resolvers.ZipResolver(assets),
        process=False,
        split_objects=True,
        group_material=False,
    )

    # positions and face count are intact, but the UV correspondence is not
    assert reloaded.area == scene.area
    with g.pytest.raises(AssertionError):
        _assert_face_records_match(
            _parse_obj_face_records(obj), _scene_face_records(reloaded)
        )


def test_obj_semantics_detect_cross_object_stitch():
    # corrupt by moving the second object's faces onto the first object's
    # positions: the per-face position tuples change and must be detected
    obj, mtl = _material_switch_obj()
    scene = g.trimesh.load_scene(
        g.trimesh.util.wrap_as_stream(obj),
        file_type="obj",
        resolver=g.trimesh.resolvers.ZipResolver({"switches.mtl": mtl.encode()}),
        process=False,
        split_objects=True,
        group_material=False,
    )
    target = list(scene.geometry.values())[1]
    # rewire every face onto a single in-bounds vertex so all three corners
    # now carry the wrong position tuple: face/vertex counts stay identical,
    # only the geometric correspondence breaks
    target.faces = g.np.zeros_like(target.faces)

    exported, assets = g.trimesh.exchange.obj.export_obj(
        scene, return_texture=True, include_normals=True
    )
    reloaded = g.trimesh.load_scene(
        g.trimesh.util.wrap_as_stream(exported),
        file_type="obj",
        resolver=g.trimesh.resolvers.ZipResolver(assets),
        process=False,
        split_objects=True,
        group_material=False,
    )

    with g.pytest.raises(AssertionError):
        _assert_face_records_match(
            _parse_obj_face_records(obj), _scene_face_records(reloaded)
        )


if __name__ == "__main__":
    test_no_vertices()
