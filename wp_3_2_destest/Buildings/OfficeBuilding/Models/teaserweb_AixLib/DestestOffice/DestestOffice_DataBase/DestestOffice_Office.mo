 
within teaserweb_AixLib.DestestOffice.DestestOffice_DataBase;
record DestestOffice_Office "DestestOffice_Office"
  extends AixLib.DataBase.ThermalZones.ZoneBaseRecord(
    T_start = 293.15,
    withAirCap = true,
    VAir = 3750.0,
    AZone = 1250.0,
    alphaRad = 5.0,
    lat = 0.88645272708792,
    nOrientations = 6,
    AWin = {74.35738476193754, 0.0, 74.35738476193754, 13.112763156279073, 0.0, 13.112763156279073},
    ATransparent = {74.35738476193754, 0.0, 74.35738476193754, 13.112763156279073, 0.0, 13.112763156279073},
    alphaWin = 2.7,
    RWin = 0.0009332630046258743,
    gWin = 0.78,
    UWin= 3.001782134105591,
    ratioWinConRad = 0.03,
    AExt = {223.0721542858126, 479.16666666666663, 223.0721542858126, 39.33828946883721, 479.16666666666663, 39.33828946883721},
    alphaExt = 2.053854562212101,
    nExt = 1,
    RExt = {2.1690934203885698e-05},
    RExtRem = 0.0009760982983114852 ,
    CExt = {433298456.24312246},
    AInt = 3541.666666666666,
    alphaInt = 2.229411764705883,
    nInt = 1,
    RInt = {1.4234773342855843e-05},
    CInt = {476655720.219049},
    AFloor = 0.0,
    alphaFloor = 0.0,
    nFloor = 1,
    RFloor = {0.00001},
    RFloorRem =  0.00001,
    CFloor = {0.00001},
    ARoof = 0.0,
    alphaRoof = 0.0,
    nRoof = 1,
    RRoof = {0.00001},
    RRoofRem = 0.00001,
    CRoof = {0.00001},
    nOrientationsRoof = 1,
    tiltRoof = {0.0},
    aziRoof = {0.0},
    wfRoof = {0.0},
    aRoof = 0.0,
    aExt = 0.5,
    TSoil = 286.15,
    alphaWallOut = 20.0,
    alphaRadWall = 5.0,
    alphaWinOut = 20.0,
    alphaRoofOut = 0.0,
    alphaRadRoof = 0.0,
    tiltExtWalls = {1.5707963267948966, 0.0, 1.5707963267948966, 1.5707963267948966, 0.0, 1.5707963267948966},
    aziExtWalls = {0.0, 0.0, 3.141592653589793, -1.5707963267948966, 0.0, 1.5707963267948966},
    wfWall = {0.13944904232146915, 0.3166451218151014, 0.13944904232146915, 0.024591535463299088, 0.0, 0.024591535463299088},
    wfWin = {0.4250443524541691, 0.0, 0.4250443524541691, 0.07495564754583085, 0.0, 0.07495564754583085},
    wfGro = 0.3552737226153621,
    nrPeople = 62.5,
    ratioConvectiveHeatPeople = 0.5,
    nrPeopleMachines = 87.5,
    ratioConvectiveHeatMachines = 0.75,
    lightingPower = 12.5,
    ratioConvectiveHeatLighting = 0.9,
    useConstantACHrate = false,
    baseACH = 0.2,
    maxUserACH = 1.0,
    maxOverheatingACH = {3.0, 2.0},
    maxSummerACH = {1.0, 283.15, 290.15},
    winterReduction = {0.2, 273.15, 283.15},
    withAHU = false,
    minAHU = 0.0,
    maxAHU = 2.6,
    hHeat = 45951.28753612509,
    lHeat = 0,
    KRHeat = 10000,
    TNHeat = 1,
    HeaterOn = true,
    hCool = 0,
    lCool = 0.0,
    KRCool = 10000,
    TNCool = 1,
    CoolerOn = false);
end DestestOffice_Office;
