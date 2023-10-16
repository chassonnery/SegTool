# trace generated using paraview version 5.10.0
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10


#### import the simple module from the paraview
from paraview.simple import *


#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


#### Find source to which the subsequent filters will be applied
source = GetActiveSource()


#### Create a 'Table To Points' object containing the data from the source
tableToPoints1 = TableToPoints(registrationName='Rods DataTable', Input=source, XColumn='X', YColumn='Y', ZColumn='Z')


#### Create a group dataset containing two orientation vectors for each rod, of equal direction but opposite orientation
# Create a first 'Calculator' object to compute the positive orientation vector of each rod
calculator1 = Calculator(registrationName='Calculator1', Input=tableToPoints1, Function='"wX"*iHat+"wY"*jHat+"wZ"*kHat')
# Create a second 'Calculator' object to compute the negative orientation vector of each rod
calculator2 = Calculator(registrationName='Calculator2', Input=tableToPoints1, Function='-"wX"*iHat-"wY"*jHat-"wZ"*kHat')
# Merge the two datasets in a 'Group Datasets' object
groupDatasets1 = GroupDatasets(registrationName='GroupDatasets1', Input=[calculator1, calculator2])
groupDatasets1.BlockNames = ['Calculator1', 'Calculator2']


#### Create a 'Glyph' object of type 'Arrow' to represent the data points
# Create an arrow glyph with position array X,Y,Z (input from object groupDatasets1)
glyph1 = Glyph(registrationName='Fibers View', Input=groupDatasets1, GlyphType='Arrow')
glyph1.OrientationArray = ['POINTS', 'Result']
# Scale arrow length with column 'Lfib' of the input data
glyph1.ScaleArray = ['POINTS', 'Length']
# Since each glyph represent only half of a rod, scale its length by a factor 0.5 to obtain rod half-length
glyph1.ScaleFactor = 0.5
# Display all data points
glyph1.GlyphMode = 'All Points'
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


#### Color each rod by its alignment rate
# Set scalar coloring
ColorBy(glyph1Display, ('POINTS','Alignment'))
# Get the color transfer function for 'Alfib'
AlignLUT = GetColorTransferFunction('Alignment')
# Rescale the transfert function to the full range of quantifier 'Alfib' (e.g. [0,1]) independently of the input data range
AlignLUT.RescaleTransferFunction(0.0, 1.0)


#### Display color legend with black labels over white background
# Get color bar
AlignLUTColorBar = GetScalarBar(AlignLUT, renderView1)
# Set color bar to visible
AlignLUTColorBar.Visibility = 1
# Set labels text color to black
AlignLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
# Set background color to white
renderView1.Background = [1.0, 1.0, 1.0]
