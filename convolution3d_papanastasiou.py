import numpy as np
import cv2
import skvideo
import skvideo.io
import operator


def myConv3D(A, B, param):
	'''the number of the padding taken from the function dimension(new)=[[dimension(old)-size(kernel)+2p]/stride]+1
	the stride is 1,dimension(old)=dimension(new),so p=(size-1)/2. p is the number of zeros which must be added
	in the begin and in the end of each dimension of the 3d array'''
	pd = int((param - 1) / 2)
	#create an array with the same dimensions of the A
	R=np.zeros_like(A)
	#the number which must be substracted from the dimension to find the last index of the loop
	x=param-pd
	#the variable to save the result of the convolution of the two 3x3x3 arrays
	element=0
	#loops to get every possible part 3x3x3 array of the A
	for k in range(A.shape[0] - x):
		for j in range(A.shape[1] - x):
			for i in range(A.shape[2]- x):
				C=A[k:k+param,j:j+param,i:i+param]
				#loops for the convolution of the 3x3x3 array with the kernel
				for l in range(param):
					for m in range(param):
						for n in range(param):
							#the result of the sum of every multiplication of the elements
							element+=C[l][m][n]*B[l][m][n]
				#one element of the final array which is put in the corresponding index
				#leaving the zeros for the padding
				R[k+pd][j+pd][i+pd]=element
				#intialization of the variable for the next loop
				element=0
	return R


def create_smooth_kernel(size):
	#create a 3d array with each element equal to 1/size^3
	kernel=np.array([[[1/pow(size,3) for k in range(size)] for j in range(size)] for i in range(size)])
	return kernel

def pad_image(A, size):
	'''the number of the padding taken from the mathtype: dimension(new)=[[dimension(old)-size(kernel)+2p]/stride]+1
	the stride is 1,dimension(old)=dimension(new),so p=(size-1)/2. p is the number of zeros which must be added
	 in the begin and in the end of each dimension of the 3d array'''
	pd=int((size-1)/2)
	#the dimensions of the array which will be padded with zeros
	sh=A.shape
	#the pd will be only for the one side of each dimensions,so it must be doubled
	pa=(2*pd,2*pd,2*pd)
	#add each element of the tuples
	#get the final dimensions of the padded array
	ne=tuple(map(operator.add, sh, pa))
	#create an array of zeros with the final dimensions
	pad_array= np.zeros(ne)
	#put the input array inside the array of zeros leaving zeros
	#in the beginning and in the end of each dimension
	pad_array[pd:-pd, pd:-pd,pd:-pd] = A
	#return the padded array
	return pad_array


def main():
	# Create a VideoCapture object and read from the video file
	cap = cv2.VideoCapture('video.mp4')
	#the array with the frames of the video
	frames =[]
	#read the video until it finishes
	while (cap.isOpened()):
		#take any frame step by step
		ret, frame = cap.read()
		#while there is frame
		if ret== True:
			#convert it to from BGR to GRAY
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			#add the black and white frame into the array
			frames.append(gray)
			#press the key 'q' to exit
			if cv2.waitKey(0) & 0xFF == ord('q'):
				break
		#else stop the loop
		else:
			break
	#convert the list to array
	frames = np.array(frames)
	#release the object of VideoCapture
	cap.release()
	#close all the windows
	cv2.destroyAllWindows()
	#create a kernel
	k=create_smooth_kernel(3)
	#zero padding in array(frames)
	frames=pad_image(frames, k.shape[0])
	#convolute the array with the kernel
	frames=myConv3D(frames, k, k.shape[0])
	#get the resolution of the final video
	res=(frames.shape[2],frames.shape[1])
	#create a file to save the final video into a file
	output=cv2.VideoWriter('output_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'),30, res,False)
	#change the type of the array
	frames = np.array(frames, dtype=np.uint8)
	# for each frame,save it to the file output_video.mp4
	for frame in range(frames.shape[0]):
		output.write(frames[frame])
	#release the file
	output.release()

if __name__ == "__main__":
    main()
