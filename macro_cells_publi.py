# trace generated using paraview version 5.10.0
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10


#### Import the simple module from the paraview
from paraview.simple import *


#### Disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


#### Find source to which the subsequent filters will be applied
source = GetActiveSource()


#### Create a 'Table To Points' object containing the data from the source
tableToPoints1 = TableToPoints(registrationName='Spheres DataTable', Input=source, XColumn='X', YColumn='Y', ZColumn='Z')


#### Create a 'Glyph' object of type 'Sphere' to represent the data points
# Create a sphere glyph with position array X,Y,Z (input from object tableToPoints1)
glyph1 = Glyph(registrationName='Spheres View', Input=tableToPoints1, GlyphType='Sphere')
# Scale sphere diameter with column 'R' of the input data
glyph1.ScaleArray = ['POINTS','R']
# Since column 'R' contain radius values, scale it by a factor 2 to obtain diameter
glyph1.ScaleFactor = 2.0
# Display all data points
glyph1.GlyphMode = 'All Points'
# Increase resolution for a smoother rendering
glyph1.GlyphType.ThetaResolution = 20
glyph1.GlyphType.PhiResolution = 20
# All other properties are left to their default value


#### Display the glyph in a RenderView
# Find view (if it exist) or create it
renderView1 = FindViewOrCreate('RenderView1', viewtype='RenderView')
# Set it to active view
SetActiveView(renderView1)
# Set object 'glyph1' to active source
SetActiveSource(glyph1)
# Show data in view
glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')
# Reset view to fit the input data
renderView1.ResetCamera(False)


#### Color each sphere by its cluster index using a custom categorical colormap
# Set scalar coloring with a colormap separated from the rest of the view
ColorBy(glyph1Display, ('POINTS','cluster'), separate = True)
# Get the color transfer function for 'lobule'
lobuleLUT = GetColorTransferFunction('cluster', glyph1Display, separate=True)
# Rescale the transfert function to include current data range
lobuleLUT.RescaleTransferFunctionToDataRange(True)
# Interpret cluster values as categories (in contrast to scalar range)
lobuleLUT.InterpretValuesAsCategories = 1
# Apply the categorical colormap defined from Sasha Trubetskoy's work. 
# /!\ Note that this will not work if you did not install the preset beforehand under the name 'SashaTrubetskoy'.
lobuleLUT.ApplyPreset('SashaTrubetskoy', True)


#### Display color legend with black labels over white background
# Get color bar
lobuleLUTColorBar = GetScalarBar(lobuleLUT, renderView1)
# Set color bar to visible
lobuleLUTColorBar.Visibility = 1
# Set labels text color to black
lobuleLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
# Set background color to white
renderView1.Background = [1.0, 1.0, 1.0]


