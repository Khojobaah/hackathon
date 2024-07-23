import numpy as np
import cv2
import os
from denoise1 import process_image


def split_page(page):
    chunk_size = (256, 256)
    main_size = page.shape[:2]
    chunks=[]
    chunk_grid = tuple(np.array(main_size)//np.array(chunk_size))
    extra_chunk = tuple(np.array(main_size)%np.array(chunk_size))
    for yc in range(chunk_grid[0]):
        row = []
        for xc in range(chunk_grid[1]):
            chunk = page[yc*chunk_size[0]:yc*chunk_size[0]+chunk_size[0], xc*chunk_size[1]: xc*chunk_size[1]+chunk_size[1]]
            row.append(chunk)
        if extra_chunk[1]:
            chunk = page[yc*chunk_size[0]:yc*chunk_size[0]+chunk_size[0], page.shape[1]-chunk_size[1]:page.shape[1]]
            row.append(chunk)
        chunks.append(row)
    if extra_chunk[0]:
        row = []
        for xc in range(chunk_grid[1]):
            chunk = page[page.shape[0]-chunk_size[0]:page.shape[0], xc*chunk_size[1]: xc*chunk_size[1]+chunk_size[1]]
            row.append(chunk)
        
        if extra_chunk[1]:
            chunk = page[page.shape[0]-chunk_size[0]:page.shape[0], page.shape[1]-chunk_size[1]:page.shape[1]]
            row.append(chunk)
        chunks.append(row)
        
    return chunks, page.shape[:2]

# def merge_chunks(chunks, osize):
#     extra = np.array(osize)%256
#     page = np.ones(osize)
#     for i, row in enumerate(chunks[:-1]):
#         for j, chunk in enumerate(row[:-1]):
#             page[i*256:i*256+256,j*256:j*256+256]=chunk
#         page[i*256:i*256+256,osize[1]-256:osize[1]]=chunks[i,-1]

#     if extra[0]:
#         for j, chunk in enumerate(chunks[-1][:-1]):
#             page[osize[0]-256:osize[0],j*256:j*256+256]=chunk
#         page[osize[0]-256:osize[0],osize[1]-256:osize[1]]=chunks[-1,-1]

#     else:
#         for j, chunk in enumerate(chunks[-1][:-1]):
#             page[osize[0]-256:osize[0],j*256:j*256+256]=chunk
#         page[osize[0]-256:osize[0],osize[1]-256:osize[1]]=chunks[-1,-1]

#     return page

def merge_chunks(chunks, osize):
    extra = np.array(osize) % 256
    page = np.ones((*osize, 3), dtype=chunks[0, 0].dtype)
    
    for i, row in enumerate(chunks[:-1]):
        for j, chunk in enumerate(row[:-1]):
            page[i*256:i*256+256, j*256:j*256+256, :] = chunk
        page[i*256:i*256+256, osize[1]-256:osize[1], :] = chunks[i, -1]

    if extra[0]:
        for j, chunk in enumerate(chunks[-1][:-1]):
            page[osize[0]-256:osize[0], j*256:j*256+256, :] = chunk
        page[osize[0]-256:osize[0], osize[1]-256:osize[1], :] = chunks[-1, -1]
    else:
        for j, chunk in enumerate(chunks[-1][:-1]):
            page[osize[0]-256:osize[0], j*256:j*256+256, :] = chunk
        page[osize[0]-256:osize[0], osize[1]-256:osize[1], :] = chunks[-1, -1]

    return page


def denoise(chunk):
    # chunk = chunk.reshape(1,256,256,1)/255.
    # denoised = model.predict(chunk).reshape(256,256)*255.
    denoised = process_image(chunk, 'Real_Denoising')
    # denoised = process_image(chunk, 'Gaussian_Gray_Denoising')
    return denoised

def denoise_page(page):
    os.chdir('Restormer')

    chunks, osize = split_page(page)
    chunks = np.array(chunks)
    denoised_chunks = np.ones(chunks.shape)
    for i, row in enumerate(chunks):
        for j, chunk in enumerate(row):
            denoised = denoise(chunk)
            denoised_chunks[i][j]=denoised
    denoised_page = merge_chunks(denoised_chunks, osize)

    return denoised_page



image = cv2.imread('./TT.jpg')

denoised_jba = denoise_page(image)

print(denoised_jba)

full_page = denoised_jba.astype(np.uint8)

page_bgr = cv2.cvtColor(full_page, cv2.COLOR_RGB2BGR)

# cv2.imwrite('HALLEL.jpg', denoised_jba)
cv2.imwrite('HALLEL2024.jpg', page_bgr)

print("Saved...")

