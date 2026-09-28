#!/usr/bin/env python


import string, os, sys

#### define various pathes to software packages

### providenceTemplate is just a local version of the MNI template - providenceHeadTemplate includes the skull, providenceTemplate is skull stipped

softwareHome = "/PROCESSING_SCIENTIFICSOFTWARE/"

providenceTemplate = "providenceBrain_OverallTemplate.nii.gz"
providenceHeadTemplate = "providenceHead_OverallTemplate.nii"

### path to ANTS bin directory
antsImageAlign = softwareHome+"/ANTs2.2/bin/ants/bin/WarpImageMultiTransform 3 "
antsDenoise = softwareHome+"/ANTs2.2/bin/ants/bin/DenoiseImage -d 3 -n Rician -i "

resetImageDimsions = "resetImageDimensions"

### T2 mapping code
t2mapper = "multiTE_T2mapper "

if len(sys.argv[1:]) < 1:
	print('Correct Usage: python multiEcho_dataConversionAndVolumeRecon.py <path to subject data directory> ')
	sys.exit(1)
	

input = sys.argv[1:]
sourceDirectory = input[0]+"/"

# set up the result directory
triplaneT2MappingDirectory = sourceDirectory+"triplaneT2/"
if os.path.exists(triplaneT2MappingDirectory):
	pass
else:
	
	os.system("mkdir "+triplaneT2MappingDirectory)

# define the path to the multi-echo reconstructed volume directory since our source data and transformation matrices are held there
reconstructedVolumeDirectory = sourceDirectory+"multiEchoVolumeReconstruction/"
volumeReference = reconstructedVolumeDirectory+"reconstructedVolumetemplate0.nii.gz"

# first thing, lets find the various files and warp files we need
shortTEImage_orig = reconstructedVolumeDirectory+"/T2sagshortTE_reshaped.nii"
midTEImage_orig = reconstructedVolumeDirectory+"/T2axialmidTE_reshaped.nii"
longTEImage_orig = reconstructedVolumeDirectory+"/T2coronallongTE_reshaped.nii"

shortTEImage_denoise = reconstructedVolumeDirectory+"/T2sagshortTE_denoised.nii"
midTEImage_denoise = reconstructedVolumeDirectory+"/T2axialmidTE_denoised.nii"
longTEImage_denoise = reconstructedVolumeDirectory+"/T2coronallongTE_denoised.nii"

os.system(antsDenoise+shortTEImage_orig+" -o "+shortTEImage_denoise)
os.system(antsDenoise+midTEImage_orig+" -o "+midTEImage_denoise)
os.system(antsDenoise+longTEImage_orig+" -o "+longTEImage_denoise)


# grab the warp files
for file in os.listdir(reconstructedVolumeDirectory):
	if file.find("T2sagshortTE")>=0 and file.find("1Warp.nii.gz")>=0:
		shortTEImage_warp = reconstructedVolumeDirectory+"/"+file
	elif file.find("T2axialmidTE")>=0 and file.find("1Warp.nii.gz")>=0:
		midTEImage_warp = reconstructedVolumeDirectory+"/"+file
	elif file.find("T2coronallongTE")>=0 and file.find("1Warp.nii.gz")>=0:
		longTEImage_warp = reconstructedVolumeDirectory+"/"+file
		
	elif file.find("T2sagshortTE")>=0 and file.find("Affine.mat")>=0:
		shortTEImage_lin = reconstructedVolumeDirectory+"/"+file
	elif file.find("T2axialmidTE")>=0 and file.find("Affine.mat")>=0:
		midTEImage_lin = reconstructedVolumeDirectory+"/"+file
	elif file.find("T2coronallongTE")>=0 and file.find("Affine.mat")>=0:
		longTEImage_lin = reconstructedVolumeDirectory+"/"+file
		
	else:
		pass
		
shortTEImage_aligned = triplaneT2MappingDirectory+"T2sagshortTE_aligned.nii"
midTEImage_aligned = triplaneT2MappingDirectory+"T2axialmidTE_aligned.nii"
longTEImage_aligned = triplaneT2MappingDirectory+"T2coronallongTE_aligned.nii"

os.system(antsImageAlign+" "+shortTEImage_denoise+" "+shortTEImage_aligned+" -R "+volumeReference+" "+shortTEImage_warp+" "+shortTEImage_lin+" --use-BSpline")
os.system(antsImageAlign+" "+midTEImage_denoise+" "+midTEImage_aligned+" -R "+volumeReference+" "+midTEImage_warp+" "+midTEImage_lin+" --use-BSpline")
os.system(antsImageAlign+" "+longTEImage_denoise+" "+longTEImage_aligned+" -R "+volumeReference+" "+longTEImage_warp+" "+longTEImage_lin+" --use-BSpline")

#os.system("fslchfiletype NIFTI "+shortTEImage_warped+" "+shortTEImage_aligned)
#os.system("fslchfiletype NIFTI "+midTEImage_warped+" "+midTEImage_aligned)
#os.system("fslchfiletype NIFTI "+longTEImage_warped+" "+longTEImage_aligned)


multiEchoFiles = []
multiEchoTimes = []
multiEchoFiles.append(shortTEImage_aligned)
multiEchoFiles.append(midTEImage_aligned)
multiEchoFiles.append(longTEImage_aligned)
multiEchoTimes.append("110.8")
multiEchoTimes.append("181.2")
multiEchoTimes.append("241.6")
Nechos = len(multiEchoFiles)
resultT2Map = triplaneT2MappingDirectory+"triplaneMultiEcho_T2Map.nii"
resultMoMap = triplaneT2MappingDirectory+"triplaneMultiEcho_MoMap.nii"
resultT2Map_lin = triplaneT2MappingDirectory+"triplaneMultiEcho_linT2Map.nii"
resultMoMap_lin = triplaneT2MappingDirectory+"triplaneMultiEcho_linMoMap.nii"


mappingCommand = t2mapper+" "+str(Nechos)+" "
for i in range(Nechos):
	mappingCommand+=multiEchoFiles[i]+" "+multiEchoTimes[i]+" "
mappingCommand+=resultT2Map+" "+resultMoMap+" "+resultT2Map_lin+" "+resultMoMap_lin+" "
print(mappingCommand)

os.system(mappingCommand)

# as a final step, we want to reset the orientation labels and such
# now re-orient the images
for file in os.listdir(triplaneT2MappingDirectory):
	if file.find("Map.nii")>0:
		os.system("fslcpgeom "+providenceTemplate+" "+triplaneT2MappingDirectory+"/"+file+" -d ")
		
		imageDimensions = " 0.9 0.9 0.9 "
		fileSplit = file.split(".nii")
		newfile = fileSplit[0]+"_reshaped.nii"
		os.system(resetImageDimsions+" "+triplaneT2MappingDirectory+"/"+file+" "+triplaneT2MappingDirectory+"/"+newfile+" "+imageDimensions)
		
		
		
		
		
		
		
		
		
