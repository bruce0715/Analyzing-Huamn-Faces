%Part 1 (Q1) load the faces, get IO
    for i=1:151
    if i~=104
     str=strcat('face_data/face/',sprintf('face%03d.bmp', i-1) );   %concatenates two strings that form the name of the image
     IO(:,:,i)=double(imread(str));
    end
    end
   IO(:,:,104)=[];

%make dimension reshape IO to IO2
P=size(IO,1);
Q=size(IO,2);
H=size(IO,3);
IO2=reshape(IO,P*Q,H);

%mean
M=mean(IO2,2);
M2=reshape(M,P,Q);

%show mean face
figure;
imshow(M2,[])

%Get Eigenface pca centralize the matrix itself.
k=20; %get the first k eigenvector
[V,U,D]=pca(transpose(IO2));
vec3=reshape(V(:,1:k),P,Q,k);

for i = 1:20
    subplot(4,5,i)
    imshow(vec3(:,:,i),[])
end

%calculate the loss
%get test image.
    for i=152:178
    str=strcat('face_data/face/',sprintf('face%03d.bmp', i-1) );   %concatenates two strings that form the name of the image
    TO(:,:,i-151)=double(imread(str));
    end

    reshape_TO=reshape(TO,P*Q,27);
    c_reshape_TO=bsxfun(@minus,reshape_TO,M);
    %change number of eigenvectors
    V1=V(:,1:k);
    reconstruct=V1*(V1'*c_reshape_TO);
    %add mean back
    reconstruct2=bsxfun(@plus,reconstruct,M);
    reshape_reconstruct=reshape(reconstruct2,P,Q,27);
    %plot reconstruct faces
    figure;
    for i = 1:27
        subplot(5,6,i)
        imshow(reshape_reconstruct(:,:,i),[])
    end
    %plot original faces
    figure;
    for i =1:27
        subplot(5,6,i)
        imshow(TO(:,:,i),[])
    end

    %calculate the differece 
    err=[];
    for k =1:20
        V1=V(:,1:k);
        reconstruct=V1*(V1'*c_reshape_TO);
        reconstruct2=bsxfun(@plus,reconstruct,M);
        myerr=sumsqr(reconstruct2-reshape_TO)/(P*Q*27);
        err(end+1)=myerr;
    end      
    
    figure;
    plot(1:20,err);
    grid on;

    %Part 1 (Q2)
    %read the landmark point
    for i =1:151
        if i~=104
            str=strcat('face_data/landmark_87/',sprintf('face%03d_87pt.dat', (i-1)));
            IL(:,i)=double(importdata(str));
        end
    end
  IL=IL(2:end,:);
  IL(:,104)=[];

    %calculate the mean landmark
    MIL=mean(IL,2);
    MIL2=vec2mat(MIL,2);
  
    %plot the mean face and mean landmark
    figure;
    imshow(M2,[])
    hold
    scatter(MIL2(:,1),MIL2(:,2))
    

    %conduct pca and scale eigenlandmark?
    [UIL,UID,UIV]=pca(IL');
    for i =1:149
        UIL1(:,i)=UIL(:,i)*sqrt(UIV(i));
    end
  
    %plot first 5 eigenlandmark
    UIL_5=UIL1(:,1:5);
    for i =1:5
        rsh_UIL_5(:,:,i)=vec2mat(UIL_5(:,i),2)+MIL2;
    end
    figure;
    imshow(M2,[])
    hold
    for i =1:5
    scatter(rsh_UIL_5(:,1,i),rsh_UIL_5(:,2,i),'filled');
    end   

 %load test landmark
  for i =1:27
        str=strcat('face_data/landmark_87/',sprintf('face%03d_87pt.dat', i+150));
        ILT(:,i)=double(importdata(str));
  end
  ILT=ILT(2:end,:);
  

 %reconstruct landmark
 ILT_Rec=UIL(:,1:5)*(UIL(:,1:5)'*bsxfun(@minus,ILT,MIL));
 ILT_Rec=bsxfun(@plus,ILT_Rec,MIL);
 
 %plot graph
 for i=1:27
     ILT_Rec_2(:,:,i)=vec2mat(ILT_Rec(:,i),2);
     ILT2(:,:,i)=vec2mat(ILT(:,i),2);
 end

 figure;
 for i=1:27
     subplot(5,6,i)
     imshow(TO(:,:,i),[])
     hold on;
      plot(ILT_Rec_2(:,1,i),ILT_Rec_2(:,2,i),'r.','MarkerSize',6)
      plot(ILT2(:,1,i),ILT2(:,2,i),'b.','MarkerSize',6)
 end       

    err2=[];
    for k =1:5
        UV1=UIL(:,1:k);
        reconstructU=UV1*(UV1'*bsxfun(@minus,ILT,MIL));
        reconstruct2U=bsxfun(@plus,reconstructU,MIL);
        myerr=sqrt(sumsqr(reconstruct2U-ILT)/27);
        err2(end+1)=myerr;
    end      
    
    figure;
    plot(1:5,err2);
    grid on;

    %Part1(Q3)
    %make training landmarks be x-y format
    for i=1:150
        IL_Rsp(:,:,i)=vec2mat(IL(:,i),2);
    end

    %use warpImage function to rotate image
    warning('off','all');
    for i =1:150
    new_images(:,:,i)=warpImage_new(IO(:,:,i),IL_Rsp(:,:,i),MIL2);
    end

    new_images2=reshape(new_images,256*256,150);
    new_M=mean(new_images2,2);
    new_images3=bsxfun(@minus,new_images2,new_M);
    %calculate eigen-face
    [new_V,new_U,new_D]=pca(new_images2');
    new_V1=new_V(:,1:10);
    new_reconstruct=new_V1*(new_V1'*new_images3);
    %add mean back
    reconstruct3=bsxfun(@plus,new_reconstruct,new_M);
    reshape_reconstruct=reshape(reconstruct3,256,256,150);

    %reconstruct the testing faces
    for i =1:27
    Tnew_images(:,:,i)=warpImage_new(TO(:,:,i),ILT_Rec_2(:,:,i),MIL2);
    end
    Tnew_images2=reshape(Tnew_images,256*256,27);
    Tnew_images3=bsxfun(@minus,Tnew_images2,new_M);
    Tnew_reconstruct=new_V1*(new_V1'*Tnew_images3);
    Tnew_reconstruct2=bsxfun(@plus,Tnew_reconstruct,new_M);
    Tnew_reconstruct3=reshape(Tnew_reconstruct2,256,256,27);
    
    
   

    %reconstruct landmark
    recons_10landmark=UIL(:,1:10)*(UIL(:,1:10)'*bsxfun(@minus,ILT,MIL));
    recons_10landmark_addmean=bsxfun(@plus,recons_10landmark,MIL);
    for i=1:27
        recons_10landmark_resh(:,:,i)=(vec2mat(recons_10landmark_addmean(:,i),2));
    end    

    %need to change landmark back
    for i =1:27
        myrecon_image2(:,:,i)=warpImage_new(Tnew_reconstruct3(:,:,i),MIL2,recons_10landmark_resh(:,:,i));
    end

    for i=1:27
        subplot(5,6,i)
        imshow(myrecon_image2(:,:,i),[]);
    end
    

    %plot reconstruction error
    err3=[];
    for k=1:20
         new_V01=new_V(:,1:k);
         Tnew_reconstruct0=new_V01*(new_V01'*Tnew_images3);
         Tnew_reconstruct02=bsxfun(@plus,Tnew_reconstruct0,new_M);
         Tnew_reconstruct03=reshape(Tnew_reconstruct02,256,256,27);
         for i =1:27
         myrecon_image02(:,:,i)=warpImage_new(Tnew_reconstruct03(:,:,i),MIL2,recons_10landmark_resh(:,:,i));
         end
         myrecon_image02_rsh=reshape(myrecon_image02,256*256,27);
         myerr=sumsqr(myrecon_image02_rsh-reshape_TO)/(256*256*27);
         err3(end+1)=myerr;
    end      

    
    figure;
    plot(1:20,err3);
    grid on;


    %Part1(Q4)
    %generate random number of 10 eigen-vector
    f=[];l=[];
    for j=1:20
    for i =1:10
        f(end+1)=normrnd(0,sqrt(new_D(i)));
        l(end+1)=normrnd(0,sqrt(UIV(i)));
    end
        F=new_M+new_V(:,1:10)*f';
        L=MIL+UIL(:,1:10)*l';
        F_whole(:,j)=F; %F_whole is 20 random face image at mean
        L_whole(:,j)=L; %L_whole is 20 random landmark 
        f=[];l=[];
    end
    

    %reshape F_whole and L_whole
    F_whole_rsh=reshape(F_whole,256,256,20);
    for i =1:20
        L_whole_rsh(:,:,i)=vec2mat(L_whole(:,i),2);
    end

   for i =1:20
       generitive_images(:,:,i)=warpImage_new(F_whole_rsh(:,:,i),MIL2,L_whole_rsh(:,:,i));
   end

   for i=1:20
       subplot(5,4,i);
       imshow(generitive_images(:,:,i),[]);
   end
   

 %Part 2
 %(5) load male and female faces
 for i =1:79
     if i~=58
     str=strcat('face_data/male_face/',sprintf('face%03d.bmp', i-1) );
     MALE(:,:,i)=double(imread(str));
     end
 end
 MALE(:,:,58)=[];
 
 for i=1:75
     str=strcat('face_data/female_face/',sprintf('face%03d.bmp',i-1));
     FEMALE(:,:,i)=double(imread(str));
 end
 
 
%Reduce face dimension by eigenvector compute in Q1
 %Reshape the face first, and minus the mean face of eigenvectors
 MALE_RES=bsxfun(@minus,reshape(MALE,256*256,78),M);
 FEMALE_RES=bsxfun(@minus,reshape(FEMALE,256*256,75),M);
 
 %Conduct dimension reduction, note V1 is the first 20 eigenvectors plus back mean.
 MALE2=V1'*MALE_RES;
 FEMALE2=V1'*FEMALE_RES;
 
 %check if it right
 %MALE3=reshape(MALE2,256,256,78);
 %imshow(MALE3(:,:,5),[]);
 
 %we know W=Sw^(-1)*(mu1-mu2)
 Sw=(MALE2-mean(MALE2,2))*transpose(MALE2-mean(MALE2,2))+(FEMALE2-mean(FEMALE2,2))*transpose(FEMALE2-mean(FEMALE2,2));
 u_diff=mean(MALE2,2)-mean(FEMALE2,2);
 w=inv(Sw)*u_diff;

MALE_D=MALE2'*w;FEMALE_D=FEMALE2'*w;

figure;
plot(MALE_D,'b.','MarkerSize',12);
hold on
plot(FEMALE_D,'r.','MarkerSize',12);
hold off
legend({'male','female'},'FontSize',16,'Location','eastoutside');


%test the fisher face by testing data 
for i =1:10
     str=strcat('face_data/male_face/',sprintf('face%03d.bmp', 77+i) );
     MALE_T(:,:,i)=double(imread(str));
 end
 
 for i=1:10
     str=strcat('face_data/female_face/',sprintf('face%03d.bmp',74+i));
     FEMALE_T(:,:,i)=double(imread(str));
 end
 
 %Reduce face dimension by eigenvector compute in Q1
 %Reshape the face first, and minus the mean face of eigenvectors
 MALE_T_RES=bsxfun(@minus,reshape(MALE_T,256*256,10),M);
 FEMALE_T_RES=bsxfun(@minus,reshape(FEMALE_T,256*256,10),M);
 
 MALE2_T=V1'*MALE_T_RES;
 FEMALE2_T=V1'*FEMALE_T_RES;
 
 MALE_DT=MALE2_T'*w;
 FEMALE_DT=FEMALE2_T'*w;
 
figure;
plot(MALE_DT,'b.','MarkerSize',12);
hold on
plot(FEMALE_DT,'r.','MarkerSize',12);
hold off
legend({'male','female'},'FontSize',16,'Location','eastoutside');

%precision
Male_precision=size(MALE_DT(MALE_DT>0),1)/10;
Female_precision=size(FEMALE_DT(FEMALE_DT<0),1)/10;

%calculate fisher face for geometry
for i =1:89
     if i~=58
     str=strcat('face_data/male_landmark_87/',sprintf('face%03d_87pt.txt', i-1) );
     Mlandmark(:,:,i)=table2array(readtable(str));
     end
 end
 Mlandmark(:,:,58)=[];
 
 for i=1:85
     str=strcat('face_data/female_landmark_87/',sprintf('face%03d_87pt.txt',i-1));
     Flandmark(:,:,i)=table2array(readtable(str));
 end
 

 %reshape Mlandmark and Flandmark, need to be same as Q2
 for i =1:88
 Mlandmark_rsh(:,i)=reshape(Mlandmark(:,:,i)',174,1);
 end
 for i =1:85
 Flandmark_rsh(:,i)=reshape(Flandmark(:,:,i)',174,1);
 end
 

 %calculate the fisherface for landmarks
 Mlandmark_res=bsxfun(@minus,Mlandmark_rsh,MIL);
 Flandmark_res=bsxfun(@minus,Flandmark_rsh,MIL);
 
 Mlandmark_code=UIL(:,1:10)'*Mlandmark_res;
 Flandmark_code=UIL(:,1:10)'*Flandmark_res;
 
 landmark_Sw=(Mlandmark_code-mean(Mlandmark_code,2))*(Mlandmark_code-mean(Mlandmark_code,2))'+(Flandmark_code-mean(Flandmark_code,2))*(Flandmark_code-mean(Flandmark_code,2))';
 landmark_diff=mean(Mlandmark_code,2)-mean(Flandmark_code,2);
 landmark_p2_w=inv(landmark_Sw)*landmark_diff;
 
 landmark_male_result=Mlandmark_code'*landmark_p2_w;
 landmark_female_result=Flandmark_code'*landmark_p2_w;

 %calculate the fisherface for wraped images
 %load all male and female faces.
  for i =1:89
     if i~=58
     str=strcat('face_data/male_face/',sprintf('face%03d.bmp', i-1) );
     MALE_all(:,:,i)=double(imread(str));
     end
 end
 MALE_all(:,:,58)=[];
 
 for i=1:85
     str=strcat('face_data/female_face/',sprintf('face%03d.bmp',i-1));
     FEMALE_all(:,:,i)=double(imread(str));
 end

 %warped images
 warning('off','all');
 for i=1:88
     newimage_M(:,:,i)=warpImage_new(MALE_all(:,:,i),Mlandmark(:,:,i),MIL2);
 end
 
 
 for i=1:85
     newimage_Fe(:,:,i)=warpImage_new(FEMALE_all(:,:,i),Flandmark(:,:,i),MIL2);
 end

 newimage_rsh_M=reshape(newimage_M,256*256,88);
 newimage_rsh_F=reshape(newimage_Fe,256*256,85);
 
 newimage_M_res=bsxfun(@minus,newimage_rsh_M,new_M);
 newimage_F_res=bsxfun(@minus,newimage_rsh_F,new_M);
 
 newimage_M_code=new_V1'*newimage_M_res;
 newimage_F_code=new_V1'*newimage_F_res;
 
 Sw_warped=(newimage_M_code-mean(newimage_M_code,2))*(newimage_M_code-mean(newimage_M_code,2))'+(newimage_F_code-mean(newimage_F_code,2))*(newimage_F_code-mean(newimage_F_code,2))';
 u_warped_diff=mean(newimage_M_code,2)-mean(newimage_F_code,2);
 w_warped=(Sw_warped)\u_warped_diff;
 
 warpednewimage_M_result=newimage_M_code'*w_warped;
 warpednewimage_F_result=newimage_F_code'*w_warped;

 %plot the graph
 figure;
 plot(warpednewimage_M_result,landmark_male_result,'b.','MarkerSize',12);
 hold on
 plot(warpednewimage_F_result,landmark_female_result,'r.','MarkerSize',12);
 hold off
 legend({'male','female'},'FontSize',16,'Location','eastoutside');
