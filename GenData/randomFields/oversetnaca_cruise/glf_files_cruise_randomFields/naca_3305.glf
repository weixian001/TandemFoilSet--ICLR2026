# ===============================================
# NACA 4-SERIES AIRFOIL GENERATOR AND 
# BOUNDARY LAYER MESH GENERATOR
# ===============================================
# Written by Travis Carrigan
#
# v1: Jan 14, 2011
# v2: Sep 21, 2011
# v3: Oct 06, 2011
#
# Updated by Wei Xian LIM
# v1: Aug 26, 2023


package require PWI_Glyph 2.4
pw::Script loadTk
wm title . "Airfoil Mesh Generator"
set labelWidth 20
set entryWidth 10
set buttonWidth 10
grid [ttk::frame .geom -padding "5 5 5 5"] -column 0 -row 0 -sticky nwes
set naca 3305
set imported 0
set fname "Browse for airfoil data file..."

grid [labelframe .geom.lf -text "1. Generate or Import Geometry" -font {-slant italic} -padx 5 -pady 5]
grid [ttk::label .geom.lf.nacal -text "NACA 4-Series Airfoil" -width $labelWidth] -column 0 -row 0 -sticky w
grid [ttk::entry .geom.lf.nacae -width $entryWidth -textvariable naca] -column 1 -row 0 -sticky e
grid [ttk::button .geom.lf.gob -text "Create" -width $buttonWidth -command {
    if {$imported} {
        set fname "Browse for airfoil data file..."
        .geom.lf.browse configure -text fname
    }
    cleanGeom
    airfoilGen
}] -column 2 -row 0 -sticky e
grid [ttk::button .geom.lf.geomb -text "Browse" -width $buttonWidth -command {
    set types {
        {{Segment Files} {.dat} }
        {{IGES Files}    {.igs} }
        {{IGES Files}    {.iges}}
        {{DBA Files}     {.dba} }
        {{All Files}     *      }
    }
    set fname [tk_getOpenFile -title "Select Airfoil Segment File" -filetypes $types]
    if {[file readable $fname]} {
        set entryWidthBrowse [expr $labelWidth+$entryWidth+1]
      set fileLength [string length $fname]
        set xv [expr $fileLength-$entryWidthBrowse]
        if {$xv<0} {set xv 0}
        .geom.lf.browse xview $xv
        cleanGeom
        pw::Database import $fname
      set imported 1
    } else {
        puts "Can't read segment file."
        set fname "Browse for airfoil data file..."
        .geom.lf.browse configure -text fname
      set imported 0
    }
}] -column 2 -row 1 -sticky e
grid configure [entry .geom.lf.browse -width [expr $labelWidth+$entryWidth+1] -text fname] -columnspan 2 -row 1 -sticky e

# Default boundary layer parameters
# cellgr: 1.05 for 1e-5, 1.03 for 1e-4
set initds 2e-5
set cellgr 1.1
set bldist 0.3
set numpts [expr int(1.0*800)]

# Create mesh information frame
grid [ttk::frame .mesh -padding "5 5 5 5"] -column 0 -row 1 -sticky nwes

grid [labelframe .mesh.lf2 -text "2. Define Boundary Layer Parameters" -font {-slant italic} -padx 5 -pady 5]
grid [ttk::label .mesh.lf2.initdsl -text "Initial Cell Height" -width $labelWidth] -column 0 -row 0 -sticky w
grid [ttk::entry .mesh.lf2.initdse -width $entryWidth -textvariable initds] -column 1 -row 0 -sticky e
grid [ttk::label .mesh.lf2.cellgrl -text "Cell Growth Rate" -width $labelWidth] -column 0 -row 1 -sticky w
grid [ttk::entry .mesh.lf2.cellgre -width $entryWidth -textvariable cellgr] -column 1 -row 1 -sticky e
grid [ttk::label .mesh.lf2.numlayerl -text "Boundary Layer Height" -width $labelWidth] -column 0 -row 2 -sticky w
grid [ttk::entry .mesh.lf2.numlayere -width $entryWidth -textvariable bldist] -column 1 -row 2 -sticky e
grid [ttk::label .mesh.lf2.cellarl -text "Points Around Airfoil" -width $labelWidth] -column 0 -row 3 -sticky w
grid [ttk::entry .mesh.lf2.cellare -width $entryWidth -textvariable numpts] -column 1 -row 3 -sticky e
grid [ttk::button .mesh.lf2.gob -text "Mesh" -width $buttonWidth -command {cleanGrid; airfoilMesh}] -column 2 -row 3 -sticky e

# Close GUI
grid [ttk::frame .close -padding "5 0 5 5"] -column 0 -row 2 -sticky nwes
grid anchor .close e
grid [ttk::button .close.gob -text "Close" -width $buttonWidth -command exit] -column 0 -row 0 -sticky e

foreach w [winfo children .geom] {grid configure $w -padx 5 -pady 5}
foreach w [winfo children .geom.lf] {grid configure $w -padx 5 -pady 5}
foreach w [winfo children .mesh] {grid configure $w -padx 5 -pady 5}
foreach w [winfo children .mesh.lf2] {grid configure $w -padx 5 -pady 5}
foreach w [winfo children .close] {grid configure $w -padx 17 -pady 0}
focus .geom.lf.nacae
::tk::PlaceWindow . widget

# AIRFOIL GENERATION PROCEDURE
# -----------------------------------------------
proc airfoilGen {} {

set m [expr {[string index $::naca 0]/100.0}]  
set p [expr {[string index $::naca 1]/10.0}] 
set a [string index $::naca 2]
set b [string index $::naca 3]
set c "$a$b"
scan $c %d c
set t [expr {$c/100.0}]

set x {}
set xu {}
set xl {}
set yu {}
set yl {}
set yc {0}
set yt {}

set ds 0.0005

if {$m == 0 && $p == 0 || $m == 0 || $p == 0} {set symm 1} else {set symm 0}


for {set i 0} {$i < [expr {1+$ds}]} {set i [expr {$i+$ds}]} {lappend x $i}

foreach xx $x {

  # Mean camber line definition for symmetric geometry
  if {$symm == 1} {lappend yc 0}

  # Mean camber line definition for cambered geometry
  if {$symm == 0 && $xx <= $p} {
    lappend yc [expr {($m/($p**2))*(2*$p*$xx-$xx**2)}]
  } elseif {$symm == 0 && $xx > $p} {
    lappend yc [expr {($m/((1-$p)**2)*(1-2*$p+2*$p*$xx-$xx**2))}]
  }

  # Thickness distribution
  lappend yt [expr {($t/0.20)*(0.29690*sqrt($xx)-0.12600*$xx- \
                    0.35160*$xx**2+0.28430*$xx**3-0.10150*$xx**4)}]

  # Theta
  set dy [expr {[lindex $yc end] - [lindex $yc end-1]}]
  set th [expr {atan($dy/$ds)}]

  # Upper x and y coordinates
  lappend xu [expr {1.0*($xx-[lindex $yt end]*sin($th))}]
  lappend yu [expr {1.0*([lindex $yc end]+[lindex $yt end]*cos($th))}]

  # Lower x and y coordinates
  lappend xl [expr {1.0*($xx+[lindex $yt end]*sin($th))}]
  lappend yl [expr {1.0*([lindex $yc end]-[lindex $yt end]*cos($th))}]

}


set airUpper [pw::Application begin Create]
set airUpperPts [pw::SegmentSpline create]

for {set i 0} {$i < [llength $x]} {incr i} {
  $airUpperPts addPoint [list [lindex $xu $i] -0.5 [lindex $yu $i]]
}

set airUpperCurve [pw::Curve create]
$airUpperCurve addSegment $airUpperPts
$airUpper end

# Create lower airfoil surface
set airLower [pw::Application begin Create]
set airLowerPts [pw::SegmentSpline create]

for {set i 0} {$i < [llength $x]} {incr i} {
  $airLowerPts addPoint [list [lindex $xl $i] -0.5 [lindex $yl $i]]
}

set airLowerCurve [pw::Curve create]
$airLowerCurve addSegment $airLowerPts
$airLower end

# Create flat trailing edge
set airTrail [pw::Application begin Create]
set airTrailPts [pw::SegmentSpline create]
$airTrailPts addPoint [list [lindex $xu end] -0.5 [lindex $yu end]]
$airTrailPts addPoint [list [lindex $xl end] -0.5 [lindex $yl end]]
set airTrailCurve [pw::Curve create]
$airTrailCurve addSegment $airTrailPts
$airTrail end

# Zoom to airfoil
pw::Display resetView +y

}



proc airfoilMesh {} {


set initDs $::initds
set cellGr $::cellgr
set blDist $::bldist
set numPts $::numpts


set dbEnts [pw::Database getAll]

# Get the curve length of all db curves
foreach db $dbEnts {
    lappend crvLength [$db getLength 1.0]
}

# Find trailing edge from minimum curve length
if {[lindex $crvLength 0] < [lindex $crvLength 1]} {
    set min 0
} else {
    set min 1
}

if {[lindex $crvLength $min] < [lindex $crvLength 2]} {
    set min $min
} else {
    set min 2
}

set dbTe [lindex $dbEnts $min]

# Get upper and lower surfaces
foreach db $dbEnts {
    if {$db != $dbTe} {
        lappend upperLower $db
    }
}

# Find y values at 50 percent length of upper and lower surfaces
set y1 [lindex [[lindex $upperLower 0] getXYZ -arc 0.5] 1]
set y2 [lindex [[lindex $upperLower 1] getXYZ -arc 0.5] 1]

# Determine upper and lower surface db entities
if {$y1 < $y2} {
    set dbLower [lindex $upperLower 0]
    set dbUpper [lindex $upperLower 1]
} else {
    set dbLower [lindex $upperLower 1]
    set dbUpper [lindex $upperLower 0]
}

# Create connectors on database entities
set upperSurfCon [pw::Connector createOnDatabase $dbUpper]
set lowerSurfCon [pw::Connector createOnDatabase $dbLower]
set trailSurfCon [pw::Connector createOnDatabase $dbTe]
set cons "$upperSurfCon $lowerSurfCon $trailSurfCon"

# Calculate main airfoil connector dimensions
foreach con $cons {lappend conLen [$con getLength -arc 1]}
set upperSurfConLen [lindex $conLen 0]
set lowerSurfConLen [lindex $conLen 1]
set trailSurfConLen [lindex $conLen 2]
set conDim [expr int($numPts/2)]

# Dimension upper and lower airfoil surface connectors
$upperSurfCon setDimension $conDim
$lowerSurfCon setDimension $conDim

# Dimension trailing edge airfoil connector
set teDim [expr int($trailSurfConLen/(10*$initDs))+2]
$trailSurfCon setDimension $teDim

# Set leading and trailing edge connector spacings
# 
set ltDs [expr 10*$initDs]

set upperSurfConDis [$upperSurfCon getDistribution 1]
set lowerSurfConDis [$lowerSurfCon getDistribution 1]
set trailSurfConDis [$trailSurfCon getDistribution 1]

$upperSurfConDis setBeginSpacing $ltDs
$upperSurfConDis setEndSpacing $ltDs
$lowerSurfConDis setBeginSpacing $ltDs
$lowerSurfConDis setEndSpacing $ltDs

# Create edges for structured boundary layer extrusion
set afEdge [pw::Edge createFromConnectors -single $cons]
set afDom [pw::DomainStructured create]
$afDom addEdge $afEdge

# Extrude boundary layer using normal hyperbolic extrusion method
set afExtrude [pw::Application begin ExtrusionSolver $afDom]
  $afDom setExtrusionSolverAttribute NormalInitialStepSize $initDs
  $afDom setExtrusionSolverAttribute SpacingGrowthFactor $cellGr
  $afDom setExtrusionSolverAttribute NormalMarchingVector {0 1 0}
  $afDom setExtrusionSolverAttribute NormalKinseyBarthSmoothing 3
  $afDom setExtrusionSolverAttribute NormalVolumeSmoothing 0.3
  $afDom setExtrusionSolverAttribute StopAtHeight $blDist
  $afExtrude run 1000
$afExtrude end

# Extrude into 3D mesh
set _TMP(mode_1) [pw::Application begin Create]
  set _DM(1) [pw::GridEntity getByName dom-1]
  set _TMP(PW_1) [pw::FaceStructured createFromDomains [list $_DM(1)]]
  set _TMP(face_1) [lindex $_TMP(PW_1) 0]
  unset _TMP(PW_1)
  set _BL(1) [pw::BlockStructured create]
  $_BL(1) addFace $_TMP(face_1)
$_TMP(mode_1) end
unset _TMP(mode_1)
set _TMP(mode_1) [pw::Application begin ExtrusionSolver [list $_BL(1)]]
  $_TMP(mode_1) setKeepFailingStep true
  $_BL(1) setExtrusionSolverAttribute Mode Translate
  $_BL(1) setExtrusionSolverAttribute TranslateDirection {1 0 0}
  $_BL(1) setExtrusionSolverAttribute TranslateDirection {0 1 0}
  $_BL(1) setExtrusionSolverAttribute TranslateDistance 1
  $_TMP(mode_1) run 1
$_TMP(mode_1) end
unset _TMP(mode_1)
unset _TMP(face_1)
pw::Application markUndoLevel {Extrude, Translate}

pw::Application setCAESolver OpenFOAM 3
pw::Application markUndoLevel {Select Solver}

set _TMP(PW_1) [pw::BoundaryCondition create]
pw::Application markUndoLevel {Create BC}

unset _TMP(PW_1)
set _TMP(PW_1) [pw::BoundaryCondition create]
pw::Application markUndoLevel {Create BC}

unset _TMP(PW_1)
set _TMP(PW_1) [pw::BoundaryCondition create]
pw::Application markUndoLevel {Create BC}

unset _TMP(PW_1)
set _TMP(PW_1) [pw::BoundaryCondition create]
pw::Application markUndoLevel {Create BC}

unset _TMP(PW_1)
set _TMP(PW_1) [pw::BoundaryCondition getByName bc-2]
$_TMP(PW_1) setName oversetPatch
pw::Application markUndoLevel {Name BC}

set _TMP(PW_2) [pw::BoundaryCondition getByName bc-3]
$_TMP(PW_2) setName Airfoil
pw::Application markUndoLevel {Name BC}

set _TMP(PW_3) [pw::BoundaryCondition getByName bc-4]
$_TMP(PW_3) setName Back
pw::Application markUndoLevel {Name BC}

set _TMP(PW_4) [pw::BoundaryCondition getByName bc-5]
$_TMP(PW_4) setName Front
pw::Application markUndoLevel {Name BC}

$_TMP(PW_3) setPhysicalType -usage CAE patch
pw::Application markUndoLevel {Change BC Type}

$_TMP(PW_3) setPhysicalType -usage CAE empty
pw::Application markUndoLevel {Change BC Type}

$_TMP(PW_4) setPhysicalType -usage CAE empty
pw::Application markUndoLevel {Change BC Type}

$_TMP(PW_2) setPhysicalType -usage CAE patch
pw::Application markUndoLevel {Change BC Type}

$_TMP(PW_2) setPhysicalType -usage CAE wall
pw::Application markUndoLevel {Change BC Type}

$_TMP(PW_1) setPhysicalType -usage CAE patch
pw::Application markUndoLevel {Change BC Type}

$_TMP(PW_3) apply [list [list $_BL(1) $_DM(1)]]
pw::Application markUndoLevel {Set BC}

set _DM(2) [pw::GridEntity getByName dom-6]
$_TMP(PW_1) apply [list [list $_BL(1) $_DM(2)]]
pw::Application markUndoLevel {Set BC}

set _DM(3) [pw::GridEntity getByName dom-8]
$_TMP(PW_4) apply [list [list $_BL(1) $_DM(3)]]
pw::Application markUndoLevel {Set BC}

set _DM(4) [pw::GridEntity getByName dom-4]
set _DM(5) [pw::GridEntity getByName dom-2]
set _DM(6) [pw::GridEntity getByName dom-3]
set _TMP(PW_5) [pw::BoundaryCondition getByName Unspecified]
$_TMP(PW_2) apply [list [list $_BL(1) $_DM(4)] [list $_BL(1) $_DM(5)] [list $_BL(1) $_DM(6)]]
pw::Application markUndoLevel {Set BC}

unset _TMP(PW_5)
unset _TMP(PW_1)
unset _TMP(PW_2)
unset _TMP(PW_3)
unset _TMP(PW_4)
pw::Display resetView +Y
set _DM(7) [pw::GridEntity getByName dom-5]
set _TMP(mode_1) [pw::Application begin CaeExport [pw::Entity sort [list $_BL(1) $_DM(1) $_DM(5) $_DM(6) $_DM(4) $_DM(7) $_DM(2) $_DM(3)]]]
  ##$_TMP(mode_1) initialize -strict -type CAE {./data/naca/naca_3305_1.0/}
  $_TMP(mode_1) initialize -strict -type CAE {./data/naca/naca_3305/}
  $_TMP(mode_1) verify
  $_TMP(mode_1) write
$_TMP(mode_1) end
unset _TMP(mode_1)

# Reset view
pw::Display resetView +y

}

# PROCEDURE TO DELETE ANY EXISTING GRID ENTITIES
# -----------------------------------------------
proc cleanGrid {} {

    set grids [pw::Grid getAll -type pw::Connector]

    if {[llength $grids]>0} {
        foreach grid $grids {$grid delete -force}
    }

}

# PROCEDURE TO DELETE ANY EXISTING GEOMETRY
# -----------------------------------------------
proc cleanGeom {} {

    cleanGrid    

    set dbs [pw::Database getAll]
    
    if {[llength $dbs]>0} {
        foreach db $dbs {$db delete -force}
    }

}

airfoilGen
cleanGrid
airfoilMesh
##pw::Application save  "./data/pw/naca_3305_1.0.pw"
pw::Application save  "./data/pw/naca_3305.pw"
pw::Application exit




