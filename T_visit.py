import os
import xml.etree.ElementTree as ET

import visit 
from visit_utils import EvalLinear


def read_View3DAttributes(fsession):

    # Create the View3DAttributes object
    v = visit.View3DAttributes()

    # Parse the XML file
    tree = ET.parse(fsession)
    root = tree.getroot()

    # Find the 'View3DAttributes' section
    view3d_element = root.find(".//Object[@name='View3DAttributes']")
    if view3d_element is None:
        raise ValueError("View3DAttributes section not found in the file!")

    # Iterate over all 'Field' elements and set values on 'v'
    for field in view3d_element.findall("Field"):
        name = field.attrib['name']
        value = field.text.strip()

        # Convert value based on type
        if 'Array' in field.attrib['type']:
            value = tuple(float(v) for v in value.split())
        elif field.attrib['type'] == 'double':
            value = float(value)
        elif field.attrib['type'] == 'bool':
            value = value.lower() == 'true'  # Convert to boolean

        # Set the attribute on 'v'
        setattr(v, name, value)

    return v


def read_WindowImageSize(fsession):

    # Parse the XML file
    tree = ET.parse(fsession)
    root = tree.getroot()

    # Find the 'windowImageSize' field within the 'ViewerWindow' section
    viewer_window = root.find(".//Object[@name='ViewerWindow']")
    if viewer_window is None:
        raise ValueError("ViewerWindow section not found!")

    window_image_size = viewer_window.find("Field[@name='windowImageSize']")
    if window_image_size is None:
        raise ValueError("windowImageSize field not found!")

    # Convert the text content to width and height
    width, height = map(int, window_image_size.text.split())
    return width, height


def read_Istart_Iend(fsession):
    # Parse the XML file
    tree = ET.parse(fsession)
    root = tree.getroot()

    # Find the 'indices' field within the 'databaseKeyframes' section
    databaseKeyframes = root.find(".//Object[@name='databaseKeyframes']")
    if databaseKeyframes is None:
        raise ValueError("databaseKeyframes section not found!")

    # Find the 'indices' field within the 'AttributeSubjectMap' section
    AttributeSubjectMap = databaseKeyframes.find(".//Object[@name='AttributeSubjectMap']")
    if AttributeSubjectMap is None:
        raise ValueError("AttributeSubjectMap section not found!")

    indices_field = AttributeSubjectMap.find("Field[@name='indices']")
    if indices_field is None:
        raise ValueError("indices field not found!")

    # Extract the text content and convert it to integers
    Istart, Iend = map(int, indices_field.text.split())
    return Istart, Iend


def save_WindowImage(casename, index, width, height, outdir='./'):

    # Ensure the output directory exists
    os.makedirs(outdir, exist_ok=True)

    s = visit.SaveWindowAttributes()

    s.family = 0
    s.fileName = casename + '_' + str(index).zfill(5)
    s.outputToCurrentDirectory=False
    s.outputDirectory=outdir
    s.format = s.JPEG
    s.progressive = 10
    s.quality = 90
    s.width = width
    s.height = height
    s.screenCapture= 0
    s.resConstraint = s.NoConstraint
    visit.SetSaveWindowAttributes(s)
    visit.SaveWindow()


def animate(casename, Sview, Tview,          \
            Sattr=None, Tattr=[0],           \
            Istart=None, Iend=None, Istep=1, \
            outdir='./'):

    # Set the default values of inputs
    if Sattr is None:
        Sattr = [Sview[0]]

    if Istart is None:
        Istart, _ = read_Istart_Iend(Sattr[0])

    if Iend is None:
        _, Iend = read_Istart_Iend(Sattr[0])

    # Set the views
    views = []
    for i in range(len(Sview)):
        views.append(read_View3DAttributes(Sview[i]))

    # Read the WindowImageSize
    width, height = read_WindowImageSize(Sattr[0])

    # Initial the stage
    k = 0
    visit.RestoreSession(Sattr[k], 0)

    # Main loop
    for i in range(Istart, Iend+Istep, Istep):

        if i > Iend:
            break

        # The target t
        t = float(i-Istart) / float(Iend-Istart+1)

        # Find the location of t in Tview
        for j in range(len(Tview)-1):
            if Tview[j] <= t <= Tview[j+1]:
                break

        # Interpolation
        #v = interpolate(float(t-Tview[j])/float(Tview[j+1]-Tview[j]), views[j], views[j+1])
        v = EvalLinear(float(t-Tview[j])/float(Tview[j+1]-Tview[j]) ,views[j], views[j+1])

        # Set the attributes
        if k < len(Tattr)-1:
            if t >= Tattr[k+1]:
                k = k + 1
                visit.RestoreSession(Sattr[k], 0)

        # Set the view
        visit.SetView3D(v)
        # Set the time
        visit.SetTimeSliderState(i)
        # Render
        visit.DrawPlots()

        # Save the figure
        save_WindowImage(casename, i, width, height, outdir=outdir)


