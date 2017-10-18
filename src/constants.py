img_h = 480
img_w = 640
img_size = 299
c_size = 64

img_dtype = {
	"name": "img",
	"img_h": img_h, 
	"img_w": img_w, 
	"num_c": 3,
	"cmp_h": img_size,
	"cmp_w": img_size
}

pnt_dtype = {
	"name": "pnt",
	"img_h": img_h, 
	"img_w": img_w,
	"num_c": 1,
	"cmp_h": c_size,
	"cmp_w": c_size
}

aud_dtype = {	
	"name": "aud",
	"img_h": img_h, 
	"img_w": img_w,
	"num_c": 1,
	"cmp_h": 128,
	"cmp_w": 8
}