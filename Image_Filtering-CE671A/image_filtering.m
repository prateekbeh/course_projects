img = imread('Lab_6_Data.jpg');

img_b1 = img(:,:,1);
img_b2 = img(:,:,2);
img_b3 = img(:,:,3);

lmin = 20;
lmax = 100;

[h,w] = size(img_b1);
for i = 1:1:h
    for j = 1:1:w
        if img_b1(i,j) <= lmin
            stretched_img_b1(i,j) = 0;
        end
        if lmin < img_b1(i,j) <= lmax
            stretched_img_b1(i,j) = ((img_b1(i,j)-lmin)*((255)/(lmax-lmin)));
        end
        if img_b1(i,j) > lmax
            stretched_img_b1(i,j) = 255;
        end
    end
end

lmin = 15;
lmax = 115;

[h,w] = size(img_b2); 

for i = 1:1:h
    for j = 1:1:w
        if img_b2(i,j) <= lmin
            stretched_img_b2(i,j) = 0;
        end
        if lmin < img_b2(i,j) <= lmax
            stretched_img_b2(i,j) = ((img_b2(i,j)-lmin)*((255)/(lmax-lmin)));
        end
        if img_b2(i,j) > lmax
            stretched_img_b2(i,j) = 255;
        end
    end
end

%imhist(img);
lmin = 10;
lmax = 120;

[ h,w] = size(img_b3); 

for i = 1:1:h
    for j = 1:1:w
        if img_b3(i,j) <= lmin
            stretched_img_b3(i,j) = 0;
        end
        if lmin < img_b3(i,j) <= lmax
            stretched_img_b3(i,j) = ((img_b3(i,j)-lmin)*((255)/(lmax-lmin)));
        end
        if img_b3(i,j) > lmax
            stretched_img_b3(i,j) = 255;
        end
    end
end


%avg filter
avg_filter1(1:3,1:3) = 1/9;
filtered_avg_img1 = imfilter(stretched_img_b1,avg_filter1);

avg_filter2(1:9,1:9) = 1/9;
filtered_avg_img2 = imfilter(stretched_img_b1,avg_filter2);

%median filter
filtered_med_img1 = medfilt2(stretched_img_b1,[5,5]);
filtered_med_img2 = medfilt2(stretched_img_b1,[9,9]);

v_kernel = [-1,0,1];
h_kernel = [-1;0;1];

%hist equalization on img
h_filtered = imfilter(histeq(stretched_img_b1),h_kernel);
v_filtered = imfilter(histeq(stretched_img_b1),v_kernel);

figure(1)
subplot(2,1,1); imshow(img_b1); title('Red Band Image');
subplot(2,1,2); imshow(stretched_img_b1); title('Red Band Stretched Image');

figure(2)
subplot(2,1,1); imshow(filtered_avg_img1); title('Avg Filter on Red Band Image with kernel 3x3');
subplot(2,1,2); imshow(filtered_avg_img2); title('Avg Filter on Red Band Image with kernel 9x9');

figure(3)
subplot(2,1,1); imshow(filtered_med_img1); title('Median Filter on Red Band Image with kernel 5x5');
subplot(2,1,2); imshow(filtered_med_img2); title('Median Filter on Red Band Image with kernel 9x9');

figure(4)
subplot(2,1,1); imshow(h_filtered); title('Horizontal Pass filter');
subplot(2,1,2); imshow(v_filtered); title('Vertical Pass filter');
