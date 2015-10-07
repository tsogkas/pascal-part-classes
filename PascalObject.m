classdef PascalObject < handle
% PASCALOBJECT  A PASCAL VOC object (inherits from handle).
%   PASCALOBJECT is an object belonging to one out of twenty classes
%   of the PASCAL dataset.  
%
% Properties:
%   class:      class name  (string)
%   classIndex: class index (integer/double)
%   mask:       object mask (logical)
%   parts:      object parts (array of PascalPart objects)
%   bbox:       object bounding box (1x4 vector of doubles)
%   centroid    object centroid coordinates (double)
%   polygon     mask polygon (useful when we want to avoid keeping mask in the memory)
%   imname      name of the image containing the object (string)
%   imsize      size of the object canvas (by default equal to the mask size)
% 
%
% obj = PASCALOBJECT()  creates an empty PascalObject object
% obj = PASCALOBJECT(arg)  creates an new PascalObject object
%   and copies object information from arg. Input arg can be an
%   object class string, a struct or another PascalObject
%   object. If arg is a valid object class string, a minimal
%   PascalObject is created. If arg is a struct or a
%   PascalObject object, information from valid fields such as
%   the object bounding box, the object mask, centroids etc.,
%   are copied into the new object; otherwise, the constructor
%   tries to compute them from what information is available.
%   
% NOTE: PASCALOBJECT inherits from class handle. This makes it very 
%   efficient since all operations are performed using references and not
%   object copies. However this also means that special care should be
%   taken, especially when copying objects or passing an object as a
%   function argument. Please see the examples below for more information. 
%
% Examples:
%   obj = PASCALOBJECT();       % create empty object
%   obj = PASCALOBJECT('car');  % create empty car object
%
%   s.class = 'car'; s.mask = someBinaryMask;
%   obj1 = PASCALOBJECT(s);     % create car object from struct
%   obj2 = PASCALOBJECT(obj1);  % create new obj2 and copy field values from obj1
%                               
%   obj3 = obj1; % only copies the reference, it DOES NOT create a new object
%   obj3 == obj1 % this returns true (1). obj3 and obj1 are the same object.
%   obj3.class = 'bus';  % now obj1.class has changed too!
%   obj3 = PASCALOBJECT(obj1); 
%   obj3.class = 'bus';     % now obj3 is a new object with the same field
%                           % values as obj1, except its class.
% 
%   foo(obj1);              % if obj1 is modified in foo the changes persist!
%   foo(PASCALOBJECT(obj1)) % this way we apply foo in a new object that is
%                           % a copy of obj1 and obj1 remains unchanged
%
% See also: PascalObject.class2index, PascalObject.part2index,
%   PascalObject.mask2bbox, PascalObject.computeCentroids,
%   PascalPart.
%
% Stavros Tsogkas, <stavros.tsogkas@centralesupelec.fr>
% Last update: March 2015

    properties
        class           % class name  (string)      
        classIndex      % class index (integer)     
        mask            % object mask (logical)     
        parts           % PascalPart array               
        bbox            % object bounding box (1x4 vector)
        centroid        % object centroid coordinates
        polygon         % mask polygon TODO: make sure this get updated when reading struct
        imname          % name of the image containing the object
        imsize          % size of the object canvas (by default equal to the mask size)
    end
        
    methods
        function obj = PascalObject(arg)
            if nargin > 0
                if isstruct(arg) 
                    obj = copyFromPascalPartsStruct(obj,arg);
                elseif isa(arg, 'PascalObject')
                    obj = copyFromPascalObject(obj,arg);
                elseif ischar(arg)  % Create an empty object of given class
                    obj.class      = arg;
                    obj.classIndex = PascalObject.class2index(arg);
                else
                    error ('Input must be a struct, a PascalObject object or a string')
                end
            end
        end
        
        % Utility functions            %TODO: add method to merge part labels
        function b = mask2bbox(obj)    %TODO: Maybe make method protected
            % MASK2BBOX  Computes the bounding box of a PascalObject or 
            %   PascalPart object, from its mask.
            %
            %   b = MASK2BBOX(obj)
            [y,x] = find(obj.mask);
            b = [min(x) min(y) max(x) max(y)];            
        end     
        function m = polygon2mask(obj) %TODO: Maybe make method protected
            x = obj.polygon(1,:); y = obj.polygon(2,:);
            m = poly2mask(x, y, obj.imsize(1), obj.imsize(2));
        end 
        function p = mask2polygon(obj)
            % MASK2POLYGON Compute node coordinates of a polygon that
            %   corresponds to a binary mask. 
            %
            %   p = MASK2POLYGON(obj) returns the [x,y] node coordinates
            %   (p = [x; y])
            %
            %   NOTE: The binary mask we obtain if we use poly2mask on the
            %   p coordinates, is a (very close)  approximation of the
            %   original mask (IOU > 0.99) but not identical.
            [y,x] = find(obj.mask, 1); % find first pixel on mask boundary            
            p = bwtraceboundary(obj.mask, [y x], 'N');     
            p = fliplr(p(1:2:end,:))'; % subsample and flip x-y coordinates
        end
        function b = getbbox(obj)
            % GETBBOX  Returns the bounding box of PascalObject or PascalPart.
            %
            %   b = GETBBBOX(obj) If obj.bbox is empty, then GETBBOX 
            %       computes the bounding box from the object mask.
            %
            % See also: PascalObject.mask2bbox
            if isempty(obj.bbox)
                b = mask2bbox(obj);
            else
                b = obj.bbox;
            end
        end
        function m = getMask(obj)
            if ~isempty(obj.mask)
                m = obj.mask;
            elseif ~isempty(obj.polygon)
                m = polygon2mask(obj);
            else
                m = [];
            end
        end
        function m = getCroppedMask(obj, bb, pad)
            % GETCROPPEDMASK  Returns cropped object mask
            %
            %   m = GETCROPPEDMASK(obj)  returns the object mask, tightly
            %       cropped around the object bounding box.
            %
            %   m = GETCROPPEDMASK(obj,bb)  returns the object mask, tightly
            %       cropped around bounding box bb. bb is a four-element vector
            %       in the form: [xmin,ymin,xmax,ymax].
            %
            %   m = GETCROPPEDMASK(obj,bb,pad) also zero-pads the cropped mask
            %
            % See also: PascalObject.getbbox, padarray
            if nargin < 3, pad = 0; end
            if nargin < 2, bb  = getbbox(obj); end
            m = obj.mask(bb(2):bb(4), bb(1):bb(3));
            if pad, m = padarray(m, [pad pad]); end
        end
        function p = getPartNames(obj)
            % GETPARTNAMES Returns cell array with the part names of a PascalObject.
            %
            %   p = GETPARTNAMES(obj) 
            p = {obj.parts(:).class};
        end
        function p = hasParts(obj)
            % HASPARTS Returns true if object has parts, false otherwise.
            %
            %   p = HASPARTS(obj) returns a Nx1 logical array, where N is the
            %       number of input objects and p(i) is true if obj(i) has
            %       parts and false otherwise.
            p = false(size(obj));
            for i=1:length(obj)
                p(i) = ~isempty(obj(i).parts);
            end
        end
        
        % Matching and alignment
        function [partsRef,partsTest] = matchParts(objRef,objTest) 
            % MATCHPARTS Find part correspondences between two objects.
            %
            % [partsRef,partsTest] = MATCHPARTS(objRef,objTest) takes two
            %   PascalObject objects, matches their common parts and 
            %   discards the rest. For parts that appear multiple times in
            %   an object (e.g. wheels,windows), matching is performed in a
            %   heuristic way, by minimizing the sum of squared distances 
            %   for all permutations of part correspondences. This assumes
            %   that we have a reasonable number of instances for each part,
            %   typically < 10-15.
            %
            % See also: perms, strfind, strtok, PascalObject.getCentroids, 
            %   PascalObject.computeCentroids
            
            % we need unnormalized centroids
            if isempty(objRef.centroid) || any(objRef.centroid <= 1)
                computeCentroids(objRef,0); 
            end
            if isempty(objTest.centroid) || any(objTest.centroid <= 1)
                computeCentroids(objTest,0);
            end
            
            
            % find mutual, single and multiple parts (wrt reference object) 
            partsRef      = objRef.parts;
            partsTest     = objTest.parts;
            partRefNames  = {partsRef(:).class};
            partTestNames = {partsTest(:).class};
            isMultipleRef = ~cellfun(@isempty,strfind(partRefNames,   '_'));
            isMultipleTest= ~cellfun(@isempty,strfind(partTestNames,  '_'));
            multiplePartNames = unique(strtok(partRefNames(isMultipleRef),'_'));
            [isMutualPart,indsTest] = ismember(partRefNames,partTestNames);
            indsTestMutual =  false(1,numel(partsTest)); 
            indsTestMutual(indsTest(indsTest > 0)) = true;
            singlePartsRef  = partsRef(isMutualPart & ~isMultipleRef);
            singlePartsTest = partsTest(indsTestMutual & ~isMultipleTest);
            assert(all(strcmp({singlePartsRef(:).class},{singlePartsTest(:).class})));
                       
            % Parts that appear multiple times in an object (e.g. wheel_1,
            % wheel_2, headlight_1, headlight_2 etc.), are not necessarily
            % matched correctly. We have to explicitly determine if wheel_1
            % in reference object indeed corresponds to wheel_1 in the test
            % object or if we have to re-assign it to another wheel. We do
            % this by minimizing the sum of squared distances of multiple 
            % part correspondences for all possible permutations.
            % part centroids. 
            cref  = getCentroids(objRef,1); cref = cref(2:end,:);
            ctest = getCentroids(objTest,1);ctest = ctest(2:end,:);
            indsRefMultiple  = [];
            indsTestMultiple = [];
            for i=1:numel(multiplePartNames)
                indsRef = find(~cellfun(@isempty,strfind(partRefNames, multiplePartNames{i})));
                indsTest= find(~cellfun(@isempty,strfind(partTestNames,multiplePartNames{i})));                
                if length(indsTest) >= length(indsRef)
                    p  = perms(indsTest); % get all possible pairing combinations 
                    p  = unique(p(:,1:length(indsRef)),'rows')';
                    cr = cref(indsRef,:);
                    ct = ctest(p,:);
                    d  = sum((ct-repmat(cr,[size(p,2),1,1])).^2, 2); % point to point distances
                    d  = sum(reshape(d,size(p)));  % sum of point-to-point distances
                    [~, indMinDist] = min(d);
                    if ~isempty(indMinDist) % part exists both in ref and test
                        indsRefMultiple  = [indsRefMultiple, indsRef];
                        indsTestMultiple = [indsTestMultiple, p(:,indMinDist)'];
                    end
                else
                    p  = perms(indsRef); % get all possible pairing combinations 
                    p  = unique(p(:,1:length(indsTest)),'rows')';
                    cr = cref(p,:);
                    ct = ctest(indsTest,:);
                    d  = sum((cr-repmat(ct,[size(p,2),1,1])).^2, 2); % point to point distances
                    d  = sum(reshape(d,size(p)));  % sum of point-to-point distances
                    [~, indMinDist] = min(d);
                    if ~isempty(indMinDist) % part exists both in ref and test
                        indsRefMultiple  = [indsRefMultiple, p(:,indMinDist)'];
                        indsTestMultiple = [indsTestMultiple, indsTest];
                    end
                end
            end
            partsRef  = [singlePartsRef,  partsRef( indsRefMultiple)];
            partsTest = [singlePartsTest, partsTest(indsTestMultiple)];
            assert(numel(partsRef) == numel(partsTest))
        end
        function [info,partsRef,partsTest] = alignObjects(objRef,objTest,visualize)
            % ALIGNOBJECTS  Aligns two PascalObject objects.
            %
            % [info,partsRef,partsTest] = ALIGNOBJECTS(objRef,objTest)
            %   Aligns two objects by matching their corresponding parts
            %   and finding the affine transformation that minimizes the
            %   sum of squared distances of their part centroids.
            %
            % [info,partsRef,partsTest] = ALIGNOBJECTS(objRef,objTest, visualize)
            %   also visualizes the overlap of the parts after object alignment.
            %
            % See also: PascalObject.matchParts, cp2tform, tformfwd.
            if nargin < 3, visualize = false; end
            
            % find mutual parts and their centroids
            [partsRef,partsTest] = matchParts(objRef,objTest);
            cref  = cat(1, partsRef(:).centroid);
            ctest = cat(1, partsTest(:).centroid);
            
            % Plot transformed points
            sz           = size(objRef.mask);
            info.tform   = cp2tform(ctest, cref, 'affine');
            [xt,yt]      = tformfwd(info.tform,ctest(:,1),ctest(:,2));
            info.xt      = xt;
            info.yt      = yt;
            info.d       = sum(sum((cref - [xt,yt]).^2, 2)); % sum of squared point-to-point distances
            info.iouPart = zeros(numel(partsRef),1);
            
            if visualize
                [nr,nc] = getSubplotGrid(numel(partsRef));
                figure(1); clf
                subplot(131); imshow(objRef.mask); hold on;
                plot(cref(:,1), cref(:,2),'r.','MarkerSize',5);
                title('Reference object mask')
                subplot(132); imshow(objTest.mask); hold on;
                plot(ctest(:,1), ctest(:,2),'r.','MarkerSize',5);
                title('Nearest neighbor mask')
                subplot(133); imshow(objRef.mask); hold on;
                plot(cref(:,1), cref(:,2),'r.','MarkerSize',5);
                plot(xt, yt,'g.','MarkerSize',5);
                line([cref(:,1), xt]',[cref(:,2), yt]','linewidth',1)
                title('Aligned keypoint correspondences')
                figure(2); clf;% open figure for next subplots
            end
            for i=1:numel(partsRef)
                % Transform only the boundary and then get full polygon mask.
                % We do it this way instead of directly applying the
                % transform on the binary masks because this can possibly
                % create holes in the result and affect the final IOU score.
                % If you want to change this, replace the next two lines
                % with the commented one.
                %     [py,px] = find(partsTest(i).mask);
                [py,px] = find(partsTest(i).mask,1);
                B  = bwtraceboundary(partsTest(i).mask,[py,px],'N');px=B(:,2);py=B(:,1);
                [xt,yt]    = tformfwd(info.tform,px,py);
                maskWarped = poly2mask(xt,yt,sz(1),sz(2));
                info.iouPart(i) = iou(partsRef(i).mask, maskWarped);
                if visualize
                    [pyr,pxr] = find(partsRef(i).mask);
                    subplot(nr,nc,i); imshow(objRef.mask); hold on;
                    plot(pxr, pyr,'g.','MarkerSize',5 );
                    plot(xt, yt,'r.','MarkerSize',5 );
                    title([partsTest(i).class '. IOU: ' num2str(info.iouPart(i))]); hold off;
                end
            end            
        end
        
        % Centroid functions (TODO: re-examine implementations and add polygon handling)
        function c = getCentroids(obj, normalize)   
            % GETCENTROIDS  Returns object and parts centroids.
            %
            % c = GETCENTROIDS(obj) returns the object and part centroids,
            %   and also stores them in the object. The output is a (P+1)x2 
            %   matrix, where P is the number of parts, and the first row
            %   corresponds to the object centroid (mass center). The first
            %   column are the x-coordinates and the second column the
            %   y-coordinates.
            %
            % c = GETCENTROIDS(obj, normalize) if normalize is true, the
            %   centroid coordinates are normalized with respect to the
            %   dimensions of the object bounding box (default: false).
            %
            % See also: regionprops, PascalObject.getbbox            
            if nargin < 2, normalize = false;  end
            assert(numel(obj)==1, 'Input of getCentroids must be a single object')
            if isempty(obj.centroid)
                computeCentroids(obj, normalize);
            end
            if ~isempty(obj.parts)
                c = cat(1, obj.centroid, obj.parts(:).centroid);
            else
                c = obj.centroid;
            end
            if normalize && all(c(:) > 1) % don't do anything if centroids are already normalized
                assert(isa(obj,'PascalObject'), 'Normalization is invalid for a PascalPart') 
                bb     = getbbox(obj);
                ncoeff = [bb(3)-bb(1)+1, bb(4)-bb(2)+1];
                c = bsxfun(@rdivide, bsxfun(@minus, c, bb(1:2)), ncoeff);
            end
        end
        function computeCentroids(obj, normalize)    
            % COMPUTECENTROIDS Computes object and part centroids and stores 
            %   them in the object.
            %
            %   COMPUTECENTROIDS(obj) where obj is a PascalObject object.
            %   COMPUTECENTROIDS(obj, normalize) if normalize is true, the
            %       centroid coordinates are normalized with respect to the
            %       dimensions of the object bounding box (default: false).
            %
            % See also: regionprops, PascalObject.getbbox
            if nargin < 2, normalize = false; end;            
            for j=1:numel(obj)     % for the object itself
                assert(~isempty(obj(j).mask), 'Object mask is empty')
                props = regionprops(obj(j).mask, 'centroid');
                obj(j).centroid = props.Centroid;
                if ~isempty(obj(j).parts)
                    computeCentroids(obj(j).parts);
                end
                if normalize % normalize to [0,1] wrt the object bbox
                    bb     = getbbox(obj(j));
                    offset = bb(1:2);
                    width  = bb(3)-bb(1)+1;
                    height = bb(4)-bb(2)+1;
                    ncoeff = [width, height];
                    obj(j).centroid = (obj(j).centroid - offset) ./ ncoeff;
                    for i=1:numel(obj(j).parts)
                        p = obj(j).parts(i);
                        p.centroid = (p.centroid - offset) ./ ncoeff;
                    end
                end
            end
        end        
        function plotCentroids(obj, varargin)
            % PLOTCENTROIDS Plot object and part centroids.
            %
            %   PLOTCENTROIDS(obj, 'PropertyName',PropertyValue) 
            %     plots centroids for the object and its parts on the cropped
            %     object mask. The marker for the object centroid is double 
            %     the size of those for the parts. 
            %   
            %   Properties:
            %   'Marker':     The marker used for the centroids (default: '.r')
            %   'MarkerSize': Size of the markers (default: 10).
            %
            %   See also: parseVarargin, plot, PascalObject.getCentroids
            validArgs = {'MarkerSize','Marker'};
            defaultValues = {10,'.r'};
            vmap   = parseVarargin(varargin, validArgs, defaultValues);
            c      = getCentroids(obj);
            c(:,1) = c(:,1) - obj.bbox(1) + 1;
            c(:,2) = c(:,2) - obj.bbox(2) + 1;
            imshow(getCroppedMask(obj)); 
            hold on;
            plot(c(1,1),c(1,2),vmap('Marker'),'MarkerSize',2*vmap('MarkerSize'))
            plot(c(2:end,1),c(2:end,2),vmap('Marker'),'MarkerSize',vmap('MarkerSize'))
            hold off;
        end
        
        % Resize, rotate, crop objects and their parts
        function obj = rotate(obj,theta)
            % ROTATE Rotate PascalObject object.
            %   Rotates masks for objects and their parts and
            %   (re)calculates the rotated object and part centroids.
            %
            %   obj = ROTATE(obj,theta)  obj is an array of N PascalObject 
            %       objects (os a single object) and theta the angle of
            %       rotation in degrees.
            %   
            %   WARNING!: This can be very slow if the number of objects is large.
            %
            %
            %   See also: imrotate, regionprops, PascalObject.resize,
            %     PascalObject.mask2bbox. 
            for i=1:numel(obj)
                obj(i).mask = imrotate(obj(i).mask,theta);
                obj(i).bbox = mask2bbox(obj(i).mask);
                if ~isempty(obj(i).centroid)
                    props = regionprops(obj(i).mask,'Centroid');
                    obj(i).centroid = props.Centroid;
                end
                if ~isempty(obj(i).parts)
                    rotate(obj(i).parts,theta);
                end
            end
        end
        function obj = resize(obj,scale)  
            % RESIZE  Resize PascalObject objects.
            %   Resizes masks for objects and their parts and
            %   (re)calculates the resized object and part centroids.
            %
            %   obj = RESIZE(obj,scale)  obj is an array of N PascalObject
            %       objects (or a single object) and scale is a scalar or a
            %       two-element vector (same as for imresize).
            %
            %   See also: regionprops, imresize, PascalObject.rotate, 
            %       PascalObject.mask2bbox            
            for i=1:numel(obj)
                obj(i).mask = imresize(obj(i).mask,scale,'nearest');
                obj(i).bbox = mask2bbox(obj(i).mask);
                if ~isempty(obj(i).centroid)
                    props = regionprops(obj(i).mask,'Centroid');
                    obj(i).centroid = props.Centroid;
                end
                if ~isempty(obj(i).parts)
                    resize(obj(i).parts,scale);
                end
            end            
        end
        function obj = crop(obj,box,pad)
            % CROP   Crop PascalObject objects.
            %
            %   obj = CROP(obj) crops object mask keeping only its bounding
            %       box part and adjust the bounding box and centroid
            %       coordinates for the object and its parts.
            %
            %   obj = CROP(obj,box) crops objects around a given bounding
            %       box. box can be empty ([]), in which case the object
            %       bounding box is used.
            %
            %   obj = CROP(obj,box, pad) also adds zero-padding around the
            %       cropped object and part masks.
            %
            %   See also: PascalObject.getbbox, PascalObject.getCroppedMask.
            if nargin < 3, pad = 0; end
            if nargin < 2, box = [];end
            for i=1:numel(obj)
                if isempty(box)
                    bb = getbbox(obj(i));
                else
                    bb = box;
                end
                if ~isempty(obj(i).mask)
                    obj(i).mask = getCroppedMask(obj(i), bb, pad);
                end
                if ~isempty(obj(i).centroid)
                    obj(i).centroid = obj(i).centroid - bb(1:2) + pad + 1;
                end
                if ~isempty(obj(i).parts)
                    crop(obj(i).parts, bb, pad);
                end
                obj(i).bbox = obj(i).bbox - bb([1,2,1,2]) + pad+1;
                obj(i).imsize = [bb(4)-bb(2), bb(3)-bb(1)] + 2*pad + 1;
            end                       
        end
    end
    
    methods(Static)
        function n = nClasses()
            n = 20;
        end
        function classes = validObjectClasses()
            % VALIDOBJECTCLASSES Valid object classes for Pascal.
            %
            %   classes = VALIDOBJECTCLASSES() returns a cell array with the
            %       valid names for the 20 object class categories in the
            %       PASCAL dataset. There is an additional class 'table' which
            %       is equivalent to the 'diningtable' class.
            classes = {
                'aeroplane'    ,...
                'bicycle'      ,...
                'bird'         ,...
                'boat'         ,...
                'bottle'       ,...
                'bus'          ,...
                'car'          ,...
                'cat'          ,...
                'chair'        ,...
                'cow'          ,...
                'diningtable'  ,...
                'dog'          ,...
                'horse'        ,...
                'motorbike'    ,...
                'person'       ,...
                'pottedplant'  ,...
                'sheep'        ,...
                'sofa'         ,...
                'train'        ,...
                'tvmonitor'    ,...
                'table'};   % extra class to account for annotations in [1]
        end  
        function pmap = class2parts(objectClass)
            % CLASS2PARTS  Maps object class to its parts.
            %
            %   pmap = CLASS2PARTS() returns a containers.Map that maps each
            %       Pascal object class to its parts.
            %
            %   pmap = CLASS2PARTS(objectClass) returns the parts of an object
            %       of class objectClass.
            %
            %   Parts that end in underscore (_) can exist multiple times in
            %   the same object. 
            %   -------------------------------------------------------------
            %   EACH TIME A NEW PART IS ADDED TO SOME CLASS, 
            %   PascalObject.symmetricParts() SHOULD BE UPDATED ACCORDINGLY!!
            %   -------------------------------------------------------------
            %
            %   Some guidelines and examples for the part abbreviations:
            %   leye: left eye
            %   reye: right eye
            %   fliplate: front licence plate
            %   bliplate: back licence plate
            %
            %   See also: PascalObject.validObjectClasses
            pmap = containers.Map();
            if nargin > 0
                classes = {objectClass}; 
            else
                classes = PascalObject.validObjectClasses;
            end
            for i=1:numel(classes)
                cls = classes{i};
                switch cls
                    case 'aeroplane'
                        pmap(cls) = {'body','stern','lwing','rwing','tail'};
                        for ii = 1:10 % multiple engines
                            pmap(cls) = [pmap(cls), sprintf('engine_%d', ii)]; 
                        end
                        for ii = 1:10 % multiple wheels
                            pmap(cls) = [pmap(cls), sprintf('wheel_%d', ii)];  
                        end
                    case 'bicycle'
                        pmap(cls) = {'fwheel','bwheel','saddle','handlebar','chainwheel'};
                        for ii = 1:10   % multiple headlights
                            pmap(cls) = [pmap(cls), sprintf('headlight_%d', ii)]; 
                        end
                    case 'bird'
                        pmap(cls) = {'head','leye','reye','beak','torso',...
                            'neck','lwing','rwing','lleg','lfoot','rleg',...
                            'rfoot','tail'};
                    case {'boat','chair','diningtable','sofa','table'}
                        pmap(cls) = [];    % only silhouette mask
                    case 'bottle'
                        pmap(cls) = {'cap','body'};
                    case {'bus','car'}
                        % fliplate: front licence plate
                        % bliplate: back licence plate
                        pmap(cls) = {'frontside','leftside','rightside','backside',...
                            'roofside','leftmirror','rightmirror','fliplate','bliplate'};
                        for ii=1:10
                            pmap(cls) = [pmap(cls), sprintf('door_%d',ii)];
                        end
                        for ii=1:10
                            pmap(cls) = [pmap(cls), sprintf('wheel_%d',ii)];
                        end
                        for ii=1:10
                            pmap(cls) = [pmap(cls), sprintf('headlight_%d',ii)];
                        end
                        for ii=1:20
                            pmap(cls) = [pmap(cls), sprintf('window_%d',ii)];
                        end
                    case 'cat'
                        % lfleg: left front leg
                        % lfpa:  left front paw
                        % lbpa:  left back paw
                        pmap(cls) = {'head','leye','reye','lear','rear',...
                            'nose','torso','neck','lfleg','lfpa','rfleg',...
                            'rfpa','lbleg','lbpa','rbleg','rbpa','tail'};
                    case {'cow','sheep'}
                        % lfuleg: left front upper leg
                        % lflleg: left front lower leg
                        pmap(cls) = {'head','leye','reye','lear','rear','muzzle','lhorn',...
                            'rhorn','torso','neck','lfuleg','lflleg','rfuleg',...
                            'rflleg','lbuleg','lblleg','rbuleg','rblleg','tail'};
                    case 'dog'
                        % same as cat + muzzle
                        pmap(cls) = {'head','leye','reye','lear','rear',...
                            'nose','torso','neck','lfleg','lfpa','rfleg',...
                            'rfpa','lbleg','lbpa','rbleg','rbpa','tail','muzzle'};
                    case 'horse'
                        % same as cow but with hooves instead of horns
                        % rfho: right front hoove
                        pmap(cls) = {'head','leye','reye','lear','rear',...
                            'muzzle','lfho','rfho','lbho','rbho','torso',...
                            'neck','lfuleg','lflleg','rfuleg','rflleg',...
                            'lbuleg','lblleg','rbuleg','rblleg','tail'};
                    case 'motorbike'
                        pmap(cls) = {'fwheel','bwheel','handlebar','saddle'};
                        for ii=1:10
                            pmap(cls) = [pmap(cls), sprintf('headlight_%d', ii)];
                        end
                    case 'person'
                        % lebrow: left eyebrow
                        % rlarm: right lower arm
                        pmap(cls) = {'head','leye','reye','lear','rear',...
                            'lebrow','rebrow','nose','mouth','hair','torso',...
                            'neck','llarm','luarm','lhand','rlarm','ruarm',...
                            'rhand','llleg','luleg','lfoot','rlleg','ruleg','rfoot'};
                    case 'pottedplant'
                        pmap(cls) = {'pot','plant'};
                    case 'train'
                        % hfrontside: head frontside
                        pmap(cls) = {'head','hfrontside','hleftside',...
                            'hrightside','hbackside','hroofside'};
                        for ii=1:10
                            pmap(cls) = [pmap(cls), sprintf('headlight_%d',ii)];
                        end
                        for ii=1:10
                            pmap(cls) = [pmap(cls), sprintf('coach_%d',ii)];
                        end
                        for ii=1:10 % coach front side
                            pmap(cls) = [pmap(cls), sprintf('cfrontside_%d',ii)];
                        end
                        for ii=1:10 % coach left side
                            pmap(cls) = [pmap(cls), sprintf('cleftside_%d',ii)];
                        end
                        for ii=1:10 % coach right side
                            pmap(cls) = [pmap(cls), sprintf('crightside_%d',ii)];
                        end
                        for ii=1:10 % coach back side
                            pmap(cls) = [pmap(cls), sprintf('cbackside_%d',ii)];
                        end
                        for ii=1:10 % coach roof side
                            pmap(cls) = [pmap(cls), sprintf('croofside_%d',ii)];
                        end
                    case 'tvmonitor'
                        pmap(cls) = {'screen'};
                    otherwise
                        error('Invalid Pascal object class')
                end
            end
            if nargin > 0, pmap = pmap(objectClass); end
        end
        function smap = part2symmetric()
            % PART2SYMMETRIC  Maps parts to their symmetric.
            %
            %   smap = PART2SYMMETRIC() returns a container.Map mapping each
            %       part to its symmetric. Only parts that actually have a
            %       symmetric are included and only symmetry wrt y-axis is
            %       treated (left is mapped to right and right to left).
            %
            %   Example: smap('lwing') = 'rwing' (left wing --> right wing)
            smap = containers.Map('KeyType','char','ValueType','char');
            smap('lwing') = 'rwing'; smap('rwing') = 'lwing';
            smap('leye') = 'reye'; smap('reye') = 'leye';
            smap('lleg') = 'rleg'; smap('rleg') = 'lleg';
            smap('lfoot') = 'rfoot'; smap('rfoot') = 'lfoot';
            smap('leftside') = 'rightside'; smap('rightside') = 'leftside';
            smap('leftmirror') = 'rightmirror'; smap('rightmirror') = 'leftmirror';
            smap('lfleg') = 'rfleg'; smap('rfleg') = 'lfleg';
            smap('lfpa') = 'rfpa'; smap('rfpa') = 'lfpa';
            smap('lbleg') = 'rbleg'; smap('rbleg') = 'lbleg';
            smap('lear') = 'rear'; smap('rear') = 'lear';
            smap('lhorn') = 'rhorn'; smap('rhorn') = 'lhorn';
            smap('lfuleg') = 'rfuleg'; smap('rfuleg') = 'lfuleg';
            smap('lflleg') = 'rflleg'; smap('rflleg') = 'lflleg';
            smap('lbuleg') = 'rbuleg'; smap('rbuleg') = 'lbuleg';
            smap('lfho') = 'rfho'; smap('rfho') = 'lfho';
            smap('lbho') = 'rbho'; smap('rbho') = 'lbho';
            smap('lebrow') = 'rebrow'; smap('rebrow') = 'lebrow';
            smap('llarm') = 'rlarm'; smap('rlarm') = 'llarm';
            smap('luarm') = 'ruarm'; smap('ruarm') = 'luarm';
            smap('lhand') = 'rhand'; smap('rhand') = 'lhand';
            smap('llleg') = 'rlleg'; smap('rlleg') = 'llleg';
            smap('luleg') = 'ruleg'; smap('ruleg') = 'luleg';
            smap('lfoot') = 'rfoot'; smap('rfoot') = 'lfoot';
            smap('hleftside') = 'hrightside'; smap('hrightside') = 'hleftside';
            smap('cleftside') = 'crightside'; smap('crightside') = 'cleftside';
        end
        function cmap = colormap()
            % COLORMAP A colormap taken from the PASCAL object segmentation
            %   annotations. The first entry corresponds to background 
            %   ([0 0 0])
            cmap = [ 0         0         0
                0.5020         0         0
                     0    0.5020         0
                0.5020    0.5020         0
                     0         0    0.5020
                0.5020         0    0.5020
                     0    0.5020    0.5020
                0.5020    0.5020    0.5020
                0.2510         0         0
                0.7529         0         0
                0.2510    0.5020         0
                0.7529    0.5020         0
                0.2510         0    0.5020
                0.7529         0    0.5020
                0.2510    0.5020    0.5020
                0.7529    0.5020    0.5020
                     0    0.2510         0
                0.5020    0.2510         0
                     0    0.7529         0
                0.5020    0.7529         0
                     0    0.2510    0.5020
                0.5020    0.2510    0.5020
                     0    0.7529    0.5020
                0.5020    0.7529    0.5020
                0.2510    0.2510         0
                0.7529    0.2510         0
                0.2510    0.7529         0
                0.7529    0.7529         0
                0.2510    0.2510    0.5020
                0.7529    0.2510    0.5020
                0.2510    0.7529    0.5020
                0.7529    0.7529    0.5020
                     0         0    0.2510
                0.5020         0    0.2510
                     0    0.5020    0.2510
                0.5020    0.5020    0.2510
                     0         0    0.7529
                0.5020         0    0.7529
                     0    0.5020    0.7529
                0.5020    0.5020    0.7529
                0.2510         0    0.2510
                0.7529         0    0.2510
                0.2510    0.5020    0.2510
                0.7529    0.5020    0.2510
                0.2510         0    0.7529
                0.7529         0    0.7529
                0.2510    0.5020    0.7529
                0.7529    0.5020    0.7529
                     0    0.2510    0.2510
                0.5020    0.2510    0.2510
                     0    0.7529    0.2510
                0.5020    0.7529    0.2510
                     0    0.2510    0.7529
                0.5020    0.2510    0.7529
                     0    0.7529    0.7529
                0.5020    0.7529    0.7529
                0.2510    0.2510    0.2510
                0.7529    0.2510    0.2510
                0.2510    0.7529    0.2510
                0.7529    0.7529    0.2510
                0.2510    0.2510    0.7529
                0.7529    0.2510    0.7529
                0.2510    0.7529    0.7529
                0.7529    0.7529    0.7529
                0.1255         0         0
                0.6275         0         0
                0.1255    0.5020         0
                0.6275    0.5020         0
                0.1255         0    0.5020
                0.6275         0    0.5020
                0.1255    0.5020    0.5020
                0.6275    0.5020    0.5020
                0.3765         0         0
                0.8784         0         0
                0.3765    0.5020         0
                0.8784    0.5020         0
                0.3765         0    0.5020
                0.8784         0    0.5020
                0.3765    0.5020    0.5020
                0.8784    0.5020    0.5020
                0.1255    0.2510         0
                0.6275    0.2510         0
                0.1255    0.7529         0
                0.6275    0.7529         0
                0.1255    0.2510    0.5020
                0.6275    0.2510    0.5020
                0.1255    0.7529    0.5020
                0.6275    0.7529    0.5020
                0.3765    0.2510         0
                0.8784    0.2510         0
                0.3765    0.7529         0
                0.8784    0.7529         0
                0.3765    0.2510    0.5020
                0.8784    0.2510    0.5020
                0.3765    0.7529    0.5020
                0.8784    0.7529    0.5020
                0.1255         0    0.2510
                0.6275         0    0.2510
                0.1255    0.5020    0.2510
                0.6275    0.5020    0.2510
                0.1255         0    0.7529
                0.6275         0    0.7529
                0.1255    0.5020    0.7529
                0.6275    0.5020    0.7529
                0.3765         0    0.2510
                0.8784         0    0.2510
                0.3765    0.5020    0.2510
                0.8784    0.5020    0.2510
                0.3765         0    0.7529
                0.8784         0    0.7529
                0.3765    0.5020    0.7529
                0.8784    0.5020    0.7529
                0.1255    0.2510    0.2510
                0.6275    0.2510    0.2510
                0.1255    0.7529    0.2510
                0.6275    0.7529    0.2510
                0.1255    0.2510    0.7529
                0.6275    0.2510    0.7529
                0.1255    0.7529    0.7529
                0.6275    0.7529    0.7529
                0.3765    0.2510    0.2510
                0.8784    0.2510    0.2510
                0.3765    0.7529    0.2510
                0.8784    0.7529    0.2510
                0.3765    0.2510    0.7529
                0.8784    0.2510    0.7529
                0.3765    0.7529    0.7529
                0.8784    0.7529    0.7529
                     0    0.1255         0
                0.5020    0.1255         0
                     0    0.6275         0
                0.5020    0.6275         0
                     0    0.1255    0.5020
                0.5020    0.1255    0.5020
                     0    0.6275    0.5020
                0.5020    0.6275    0.5020
                0.2510    0.1255         0
                0.7529    0.1255         0
                0.2510    0.6275         0
                0.7529    0.6275         0
                0.2510    0.1255    0.5020
                0.7529    0.1255    0.5020
                0.2510    0.6275    0.5020
                0.7529    0.6275    0.5020
                     0    0.3765         0
                0.5020    0.3765         0
                     0    0.8784         0
                0.5020    0.8784         0
                     0    0.3765    0.5020
                0.5020    0.3765    0.5020
                     0    0.8784    0.5020
                0.5020    0.8784    0.5020
                0.2510    0.3765         0
                0.7529    0.3765         0
                0.2510    0.8784         0
                0.7529    0.8784         0
                0.2510    0.3765    0.5020
                0.7529    0.3765    0.5020
                0.2510    0.8784    0.5020
                0.7529    0.8784    0.5020
                     0    0.1255    0.2510
                0.5020    0.1255    0.2510
                     0    0.6275    0.2510
                0.5020    0.6275    0.2510
                     0    0.1255    0.7529
                0.5020    0.1255    0.7529
                     0    0.6275    0.7529
                0.5020    0.6275    0.7529
                0.2510    0.1255    0.2510
                0.7529    0.1255    0.2510
                0.2510    0.6275    0.2510
                0.7529    0.6275    0.2510
                0.2510    0.1255    0.7529
                0.7529    0.1255    0.7529
                0.2510    0.6275    0.7529
                0.7529    0.6275    0.7529
                     0    0.3765    0.2510
                0.5020    0.3765    0.2510
                     0    0.8784    0.2510
                0.5020    0.8784    0.2510
                     0    0.3765    0.7529
                0.5020    0.3765    0.7529
                     0    0.8784    0.7529
                0.5020    0.8784    0.7529
                0.2510    0.3765    0.2510
                0.7529    0.3765    0.2510
                0.2510    0.8784    0.2510
                0.7529    0.8784    0.2510
                0.2510    0.3765    0.7529
                0.7529    0.3765    0.7529
                0.2510    0.8784    0.7529
                0.7529    0.8784    0.7529
                0.1255    0.1255         0
                0.6275    0.1255         0
                0.1255    0.6275         0
                0.6275    0.6275         0
                0.1255    0.1255    0.5020
                0.6275    0.1255    0.5020
                0.1255    0.6275    0.5020
                0.6275    0.6275    0.5020
                0.3765    0.1255         0
                0.8784    0.1255         0
                0.3765    0.6275         0
                0.8784    0.6275         0
                0.3765    0.1255    0.5020
                0.8784    0.1255    0.5020
                0.3765    0.6275    0.5020
                0.8784    0.6275    0.5020
                0.1255    0.3765         0
                0.6275    0.3765         0
                0.1255    0.8784         0
                0.6275    0.8784         0
                0.1255    0.3765    0.5020
                0.6275    0.3765    0.5020
                0.1255    0.8784    0.5020
                0.6275    0.8784    0.5020
                0.3765    0.3765         0
                0.8784    0.3765         0
                0.3765    0.8784         0
                0.8784    0.8784         0
                0.3765    0.3765    0.5020
                0.8784    0.3765    0.5020
                0.3765    0.8784    0.5020
                0.8784    0.8784    0.5020
                0.1255    0.1255    0.2510
                0.6275    0.1255    0.2510
                0.1255    0.6275    0.2510
                0.6275    0.6275    0.2510
                0.1255    0.1255    0.7529
                0.6275    0.1255    0.7529
                0.1255    0.6275    0.7529
                0.6275    0.6275    0.7529
                0.3765    0.1255    0.2510
                0.8784    0.1255    0.2510
                0.3765    0.6275    0.2510
                0.8784    0.6275    0.2510
                0.3765    0.1255    0.7529
                0.8784    0.1255    0.7529
                0.3765    0.6275    0.7529
                0.8784    0.6275    0.7529
                0.1255    0.3765    0.2510
                0.6275    0.3765    0.2510
                0.1255    0.8784    0.2510
                0.6275    0.8784    0.2510
                0.1255    0.3765    0.7529
                0.6275    0.3765    0.7529
                0.1255    0.8784    0.7529
                0.6275    0.8784    0.7529
                0.3765    0.3765    0.2510
                0.8784    0.3765    0.2510
                0.3765    0.8784    0.2510
                0.8784    0.8784    0.2510
                0.3765    0.3765    0.7529
                0.8784    0.3765    0.7529
                0.3765    0.8784    0.7529
                0.8784    0.8784    0.7529];
        end
        function cmap = class2color()   % TODO: consider making this private/protected
            % CLASS2COLOR  Maps each object class to a color.
            %
            %   cmap = CLASS2COLOR() maps each object class to a
            %       three-element rgb color vector. The colors span the
            %       spectrum in a way so that they are distinguishable from
            %       one-another.
            %   
            %   See also: PascalObject.validObjectClasses, mat2cell,
            %   PascalObject.colormap
            colors   = PascalObject.colormap(); % first entry is background
            classes  = PascalObject.validObjectClasses;
            nClasses = length(classes);
            cmap = containers.Map(classes, ...
                mat2cell(colors(2:nClasses+1,:), ones(nClasses,1), 3));
            cmap('table') = cmap('diningtable');
        end
        function imap = class2index(objectClass)
            % CLASS2INDEX  Maps each Pascal Object class to an integer.
            % This method can be useful because it is much easier to add
            % new classes in Pascal.validObjectClasses without changing the
            % indexes manually (as in PascalObject.getClassIndex).
            %
            %   imap = CLASS2INDEX() returns a containers.Map
            %
            %   See also: PascalObject.validObjectClasses,
            %   PascalObject.getClassIndex
            classes  = PascalObject.validObjectClasses;
            nClasses = length(classes);
            imap = containers.Map(classes, 1:nClasses);
            imap('table') = imap('diningtable');
            if nargin > 0
                assert(any(strcmp(objectClass,classes)), 'Invalid Pascal object class')
                imap = imap(objectClass);
            end
        end
        function cind = class2indexFast(objectClass)
            % CLASS2INDEXFAST Faster version of PascalObject.class2index().
            %
            %   cind = CLASS2INDEXFAST(objectClass) returns the class index
            %   of objectClass (an integer in [1,20])
            switch objectClass
                case 'aeroplane'
                    cind = 1;
                case 'bicycle'
                    cind = 2;
                case 'bird'
                    cind = 3;
                case 'boat'
                    cind = 4;
                case 'bottle'
                    cind = 5;
                case 'bus'
                    cind = 6;
                case 'car'
                    cind = 7;
                case 'cat'
                    cind = 8;
                case 'chair'
                    cind = 9;
                case 'cow'
                    cind = 10;
                case {'diningtable','table'}
                    cind = 11;
                case 'dog'
                    cind = 12;
                case 'horse'
                    cind = 13;
                case 'motorbike'
                    cind = 14;
                case 'person'
                    cind = 15;
                case 'pottedplant'
                    cind = 16;
                case 'sheep'
                    cind = 17;
                case 'sofa'
                    cind = 18;
                case 'train'
                    cind = 19;
                case 'tvmonitor'
                    cind = 20;
                otherwise
                    error('Invalid Pascal object class')
            end          
        end
        function cmap = part2class(merge)
            % PART2CLASS  Maps part name to object classes.
            %
            %   cmap = PART2CLASS(merge) returns a container.Map that maps a
            %       part to all the object classes in which it appears. If 
            %       merge is true, parts that appear multiple times in an
            %       object are merged to a single part class. E.g.: wheel_1, 
            %       wheel_2... are merged into 'wheel'.
            %
            %   See also: PascalObject.validObjectClasses,
            %      PascalObject.class2parts, PascalObject.class2subclasses.
            if nargin < 1, merge = true; end
            cmap = containers.Map();
            classParts = PascalObject.class2parts();
            classes = PascalObject.validObjectClasses();
            allPartNames = classParts.values;
            allPartNames = unique([allPartNames{:}]);
            % Create a map for each one of the part names
            for i=1:numel(allPartNames)
                for j=1:numel(classes)
                    if any(strcmp(allPartNames{i}, classParts(classes{j})))
                        if cmap.isKey(allPartNames{i})
                            cmap(allPartNames{i}) = ...
                                [cmap(allPartNames{i}), classes(j)];
                        else
                            cmap(allPartNames{i}) = classes(j);
                        end
                    end
                end
            end            
            % merge parts that appear multiple times
            if merge    % THIS WORKS BUT IT'S MESSY. NEEDS TO BE RE-WRITTEN MAYBE. CHECK STRTOK, STRSPLIT
                allPartNamesOrig = allPartNames; % backup uncropped names
                inds = strfind(allPartNames, '_');
                for i=1:numel(allPartNames)
                    if ~isempty(inds{i})
                        allPartNames{i} = allPartNames{i}(1:(inds{i}-1));
                    end
                end
                [~, ia, ic] = unique(allPartNames); 
                firstInstance = ia(diff(ia) > 1);
                if ia(end)~= ia(end-1)  % check last element
                    firstInstance(end+1) = ia(end);
                end
                ic(end+1) = ic(end);    % dummy entry for diff
                df1 = diff(ic);
                df0 = ~df1;
                otherInstances = df0 | (df1 & [0; df0(1:end-1)]);
                % store values for new 'merged' entries in cmap
                vals = cell(numel(firstInstance),1);
                for i=1:numel(firstInstance)
                    vals{i} = cmap([allPartNames{firstInstance(i)} '_1']);
                end
                % Remove map entries for multiple instance parts
                cmap.remove(allPartNamesOrig(otherInstances));
                % Add new values
                for i=1:numel(firstInstance)
                    cmap(allPartNames{firstInstance(i)}) = vals{i};
                end
            end
            
            % Add extra entries for semantic classes that correspond to parts
            smap = PascalObject.class2subclasses();
            for key = smap.keys
                vals = smap(key{1});
                if all(ismember(vals, allPartNamesOrig)) % if vals are parts
                    if ~cmap.isKey(key{1})
                        cmap(key{1}) = {}; % create new entry              
                    end
                    for val = vals
                        ind = strfind(val{1}, '_'); % look for multiple parts
                        if isempty(ind)
                            cmap(key{1}) = [cmap(key{1}) cmap(val{1})];
                        else
                            cmap(key{1}) = [cmap(key{1}) cmap(val{1}(1:ind-1))];
                        end
                    end
                    cmap(key{1}) = unique(cmap(key{1}));
                end
            end


%             cmap('wheel') = {'bus','car','aeroplane','bicycle','motorbike'};
%             cmap('wing')  = {'aeroplane', 'bird'};
%             cmap('leg')   = {'bird','cat','cow','dog','horse','person','sheep'};
%             cmap('foot')  = {'bird','person'};
        end
        function smap = class2subclasses()
            % CLASS2SUBCLASSES  Maps semantic class to its subclasses.
            %   CLASS2SUBCLASSES maps an object or part category to its
            %   subcategories. These categories can be generalizations of 
            %   parts that appear multiple times in an object 
            %   (e.g. the 'wheel' class contains 'wheel_1', 'wheel_2' etc.) 
            %   or classes that are semantically related to their  
            %   subclasses (e.g. the 'animals' class includes subclasses 
            %   'bird','cat','dog' etc)
            %
            % smap = CLASS2SUBCLASSES()  smap is a containers.Map that maps
            %   an object or part category to the subcategories it entails.
            %
            %   Example: smap('wheel') = {'wheel_1','wheel_2',...}
            %            smap('2legs') = {'bird','person'}
            %            smap('rigid') = [smap('indoor'), smap('transport')] 
            %            smap('indoor')= {'bottle','chair',...}
            smap = containers.Map();
            % High level categories
            smap('animals') = {'bird','cat','dog','horse','sheep','cow'};
            smap('pets')    = {'cat','dog'};
            smap('2wheels') = {'bicycle','motorbike'};
            smap('4wheels') = {'car','bus'};
            smap('wheeled') = [smap('2wheels'),smap('4wheels'),{'aeroplane'}];
            smap('2legs')   = {'bird','person'};
            smap('4legs')   = {'cat','dog','cow','horse','sheep'};
            smap('winged')  = {'bird','aeroplane'};
            smap('transport') = [smap('2wheels'), smap('4wheels'), ...
                {'boat','train','aeroplane'}];
            smap('indoor') = {'bottle','chair','diningtable','table','sofa',...
                'tvmonitor','pottedplant'};
            
            % Rigid-deformable
            smap('rigid') = [smap('indoor'), smap('transport')];
            smap('deformable') = [smap('animals'), 'person'];
            
            % With/without parts
            smap('noparts') = {'boat','chair','diningtable','sofa','table'};
            smap('haveParts') = setdiff(PascalObject.validObjectClasses(), ...
                smap('noparts'));
            
            % Parts (including parts that appear multiple times)
            smap('multiple') = {'engine','wheel','headlight','door','window',...
                'coach','cfrontside','cleftside','crightside','cbackside','croofside'};
            for p=smap('multiple')
                smap(p{1}) = {[p{1} '_1']}; % add first instance
                for j=2:10                  % add rest of the instances
                    smap(p{1}) = [smap(p{1}), sprintf([p{1} '_%d'],j)];
                end
            end
            for i=11:20     % complete extra entries for window (20 instead of 10)
                smap('window') = [smap('window'), sprintf('window_%d',i)];
            end
            smap('wheel')= [{'fwheel','bwheel'}, smap('wheel')]; % ADD CHAINWHEEL??
            smap('wing') = {'lwing','rwing'};
            smap('eye')  = {'leye','reye'};
            smap('ear')  = {'lear','rear'};
            smap('arm')  = {'llarm','luarm','rlarm','ruarm'};
            smap('hand') = {'lhand','rhand'};
            smap('foot') = {'lfoot','rfoot'};
            smap('paw')  = {'lfpa','rfpa','lbpa','rbpa'};
            smap('hoof') = {'lfho','rfho','lbho','rbho'};
            smap('horn') = {'lhorn','rhorn'};
            smap('leg')  = {'lleg','rleg','lfleg','rfleg','lbleg','rbleg',...
                'lfuleg','lflleg','rfuleg','rflleg','lbuleg','lblleg',...
                'rbuleg','rblleg','llleg','luleg','rlleg','ruleg'};
            smap('mirror') = {'leftmirror','rightmirror'};
            smap('liplate')= {'fliplate','bliplate'};
            smap('brow')   = {'lebrow','rebrow'};
            smap('coachside') = [smap('cfrontside'),smap('cleftside'),...
                smap('crightside'),smap('cbackside'),smap('croofside')];
            smap('vehicleside') = {'leftside','rightside'};
        end
        function imap = part2index(objectClass)
            % PART2INDEX  Maps an object part to an integer
            %   The parts are assigned to a different integer label
            %   according to their object class.
            %
            %   imap = PART2INDEX(objectClass) imap is a containers.Map and
            %       objectClass a string.
            %
            %   See also: PascalObject.class2parts
            partNames  = PascalObject.class2parts(objectClass);
            if isempty(partNames)   
                imap = [];
            else
                imap = containers.Map(partNames,1:numel(partNames));
            end
        end
        function omap = part2occlusion(objectClass)
            % PART2OCCLUSION  Occlusion ordering for an object class.
            %   PART2OCCLUSION maps each part to a 'depth' degree creating 
            %   in this way an occlusion ordering of the object parts.
            %   This means that if two or more object parts overlap, the 
            %   one with the highest occlusion degree, will occlude the others.
            %
            %   omap = PART2OCCLUSION(objectClass) objectClass is a valid
            %       object class string and omap a containers.Map.
            %
            %   Example for the 'car' object class:
            %       omap('frontside') = 1;
            %       omap('fliplate')  = 2;
            %       omap('door')      = 2;
            %       omap('wheel')     = 3;
            %
            %   The licence plate will always occlude the front side of the
            %   car (e.g. when using PascalPart.mergeMasks), even if their
            %   individual masks overlap. If the car viewpoint is such that
            %   the door, the rightside and a right wheel are all visible,
            %   the door will occlude rightside, but will will occlude
            %   both the rightside and the door.
            %
            %   See also: PascalPart.mergeMasks
            omap = containers.Map();
            switch objectClass
                case 'aeroplane'
                    omap('body')  = 1;
                    omap('stern') = 2;
                    omap('tail')  = 2;
                    omap('lwing') = 3;
                    omap('rwing') = 3;
                    for i = 1:10
                        omap(sprintf('engine_%d', i)) = 4; 
                        omap(sprintf('wheel_%d',  i)) = 4; 
                    end                    
                case  {'bicycle','motorbike'}
                    omap('fwheel')     = 1;
                    omap('bwheel')     = 1;
                    omap('saddle')     = 2;
                    omap('handlebar')  = 2;
                    omap('chainwheel') = 2;
                    for i = 1:10
                        omap(sprintf('headlight_%d', i)) = 3; 
                    end
                case 'bird'
                    omap('torso') = 1;
                    omap('neck')  = 2;
                    omap('head')  = 1;
                    omap('tail')  = 1;
                    omap('rleg')  = 1;
                    omap('lleg')  = 1;
                    omap('rfoot') = 1;
                    omap('lfoot') = 1;
                    omap('leye')  = 2;
                    omap('reye')  = 2;
                    omap('beak')  = 2;
                    omap('lwing') = 3;
                    omap('rwing') = 3;
                case {'boat','chair','diningtable','sofa','table'}
                    omap = [];
                case 'bottle'
                    omap('body') = 1;
                    omap('cap')  = 2;
                case {'bus','car'}
                    omap('frontside')   = 1;
                    omap('leftside')    = 1;
                    omap('rightside')   = 1;
                    omap('backside')    = 1;
                    omap('roofside')    = 1;
                    omap('leftmirror')  = 4;
                    omap('rightmirror') = 4;
                    omap('fliplate')    = 2;
                    omap('bliplate')    = 2;
                    for i = 1:10
                        omap(sprintf('door_%d',i))      = 2;
                        omap(sprintf('wheel_%d',i))     = 3;
                        omap(sprintf('headlight_%d',i)) = 2;
                    end
                    for i = 1:20
                        omap(sprintf('window_%d',i))    = 3;
                    end
                case {'cat','dog'}
                    omap('head')   = 1;
                    omap('leye')   = 2;                % left eye
                    omap('reye')   = 2;                % right eye
                    omap('lear')   = 2;                % left ear
                    omap('rear')   = 2;                % right ear
                    omap('nose')   = 2;
                    omap('torso')  = 1;
                    omap('neck')   = 1;
                    omap('lfleg')  = 1;               % left front leg
                    omap('rfleg')  = 1;               % right front leg
                    omap('lbleg')  = 1;               % left back leg
                    omap('rbleg')  = 1;               % right back leg
                    omap('rfpa')   = 2;               % right front paw
                    omap('lbpa')   = 2;               % left back paw
                    omap('lfpa')   = 2;               % left front paw
                    omap('rbpa')   = 2;               % right back paw
                    omap('tail')   = 1;
                    omap('muzzle') = 3;
                case  {'cow','sheep','horse'}
                    omap('head')   = 1;
                    omap('leye')   = 2;                % left eye
                    omap('reye')   = 2;                % right eye
                    omap('lear')   = 2;                % left ear
                    omap('rear')   = 2;                % right ear
                    omap('muzzle') = 3;
                    omap('lhorn')  = 2;                % left horn
                    omap('rhorn')  = 2;                % right horn
                    omap('torso')  = 1;
                    omap('neck')   = 1;
                    omap('lfuleg') = 1;               % left front upper leg
                    omap('lflleg') = 1;               % left front lower leg
                    omap('rfuleg') = 1;               % right front upper leg
                    omap('rflleg') = 1;               % right front lower leg
                    omap('lbuleg') = 1;               % left back upper leg
                    omap('lblleg') = 1;               % left back lower leg
                    omap('rbuleg') = 1;               % right back upper leg
                    omap('rblleg') = 1;               % right back lower leg
                    omap('tail')   = 1;
                    omap('lfho')   = 2;
                    omap('rfho')   = 2;
                    omap('lbho')   = 2;
                    omap('rbho')   = 2;
                case 'person'
                    omap('head')   = 1;
                    omap('leye')   = 2;                 % left eye
                    omap('reye')   = 2;                 % right eye
                    omap('lear')   = 2;                 % left ear
                    omap('rear')   = 2;                 % right ear
                    omap('lebrow') = 2;                 % left eyebrow
                    omap('rebrow') = 2;                 % right eyebrow
                    omap('nose')   = 2;
                    omap('mouth')  = 2;
                    omap('hair')   = 3;
                    
                    omap('torso')  = 1;
                    omap('neck')   = 1;
                    omap('llarm')  = 1;                 % left lower arm
                    omap('luarm')  = 1;                 % left upper arm
                    omap('rlarm')  = 1;                 % right lower arm
                    omap('ruarm')  = 1;                 % right upper arm
                    omap('lhand')  = 2;                 % left hand
                    omap('rhand')  = 2;                 % right hand
                    
                    omap('llleg')  = 1;               	% left lower leg
                    omap('luleg')  = 1;               	% left upper leg
                    omap('rlleg')  = 1;               	% right lower leg
                    omap('ruleg')  = 1;               	% right upper leg
                    omap('rfoot')  = 2;               	% right foot
                    omap('lfoot')  = 2;               	% left foot
                case 'pottedplant'
                    omap('pot')    = 1;
                    omap('plant')  = 2;
                case 'train'
                    omap('head')       = 1;
                    omap('hfrontside') = 2;                	% head front side
                    omap('hleftside')  = 2;                	% head left side
                    omap('hrightside') = 2;                	% head right side
                    omap('hbackside')  = 2;                 % head back side
                    omap('hroofside')  = 2;                	% head roof side                    
                    for i = 1:10
                        omap(sprintf('headlight_%d',i))   = 3;
                        omap(sprintf('coach_%d',i))       = 3;
                        omap(sprintf('croofside_%d', i))  = 2;   % coach roof side
                        omap(sprintf('cfrontside_%d', i)) = 2;   % coach front side
                        omap(sprintf('cleftside_%d', i))  = 2;   % coach left side
                        omap(sprintf('crightside_%d', i)) = 2;  % coach right side
                        omap(sprintf('cbackside_%d', i))  = 2;   % coach back side
                    end
                case 'tvmonitor'
                    omap('screen') = 1;
                otherwise
                    error('Object class not supported')
            end
        end
        
        %  THESE FUNCTIONS NEED TESTING TO VERIFY THEY WORK PROPERLY
        function amap = part2adjacent() % create part adjacency map 
            warning('PascalObject.part2adjacent() has not been tested!')
            amap = containers.Map();
            smap = PascalObject.class2subclasses();
            % Start by explicitly declaring adjacencies
            % Head/face
            for p=smap('eye'), amap(p{1}) = {'head'}; end
            for p=smap('ear'), amap(p{1}) = {'head'}; end
            for p=smap('brow'),amap(p{1}) = [{'head'}, smap('eye')]; end
            amap('hair')  = [{'head'}, smap('ear')];
            amap('nose')  = [{'head','mouth'}, smap('eye')];
            amap('mouth') = {'head','nose','neck'};
            amap('neck')  = {'head','torso'};
            amap('beak')  = {'head'};
            amap('muzzle')= {'mouth','nose'};
            
            %  Arms, legs, feet, tail etc.
            amap('llarm') = {'luarm'};  % arms
            amap('rlarm') = {'ruarm'};
            amap('lhand') = {'llarm'};  % hands
            amap('rhand') = {'rlarm'};
            amap('lfoot') = {'lleg','llleg'}; % feet
            amap('rfoot') = {'rleg','rlleg'};
            amap('lfpa')  = {'lfleg'};  % paws
            amap('rfpa')  = {'rfleg'};
            amap('lbpa')  = {'lbleg'};
            amap('rbpa')  = {'rbleg'};
            amap('lfho')  = {'lflleg'}; % hooves
            amap('rfho')  = {'rflleg'};
            amap('lbho')  = {'lblleg'};
            amap('rbho')  = {'rblleg'};
            amap('lleg')  = {'torso','lfoot'};  % legs
            amap('rleg')  = {'torso','rfoot'};
            amap('lflleg')= {'lfuleg'};
            amap('rflleg')= {'rfuleg'};
            amap('rblleg')= {'rbuleg'};
            amap('lblleg')= {'lbuleg'};
            amap('rbuleg')= {'rblleg','torso'};
            amap('rfuleg')= {'rflleg','torso'};
            amap('lbuleg')= {'lblleg','torso'};
            amap('lfuleg')= {'lflleg','torso'};
            amap('lhorn') = {'head'};   % horns
            amap('rhorn') = {'head'};
            amap('tail')  = {'stern','torso'};  % tail
            amap('stern') = {'tail','body'};
            
            % Wheels, wings, mirrors, windows, engines (NEEDS TO ADD TRAIN!)
            for p=smap('wing'), amap(p{1}) = {'body','engine','torso'}; end 
            for p=smap('wheel')
                amap(p{1}) = [{'body','leftside','rightside'}, smap('wing')]; 
            end
            for p=smap('door'),  amap(p{1}) = {'leftside','rightside'}; end
            for p=smap('coach'), amap(p{1}) = smap('coachside'); end 
            for p=smap('headlight') 
                amap(p{1}) = {'handlebar','head','frontside'}; 
            end
            amap('fwheel') = {'handlebar'};
            amap('bwheel') = {'saddle','chainwheel'};
            amap('saddle') = [{'chainwheel','bwheel'},smap('headlight')];
            amap('frontside') = [smap('mirror'),smap('headlight'),...
                smap('window'), {'fliplate'}];
            amap('rightside') = [smap('door'),smap('window'), {'rightmirror'}];
            amap('leftside') = [smap('door'),smap('window'), {'leftmirror'}];
            amap('backside') = [smap('mirror'),smap('headlight'),...
                smap('window'), {'bliplate'}];
            amap('hfrontside')= {'head'};
            amap('hleftside') = {'head'};
            amap('hrightside')= {'head'};
            amap('hbackside') = {'head'};
            amap('hroofside') = {'head'};
            
            % Misc
            amap('cap') = {'body'};
            amap('pot') = {'plant'};
            amap('plant') = {'pot'};
            
            % Complete adjacencies based on what has been manually assigned
            for k = amap.keys;
                key = k{1}; % for each key get its vals
                vals = amap(key);
                for i=1:numel(vals)
                    if amap.isKey(vals{i})  % val is already a key
                        if ~any(strcmp(amap(vals{i}), key)) % key is not adjacent
                            amap(vals{i}) = [amap(vals{i}), key]; % add it
                        end
                    else    % create new key from vals{i} and add key as adjacent
                        amap(vals{i}) = {key};
                    end
                end
            end
            % Chech adjacencies are symmetric
            for k = amap.keys
                key = k{1};
                vals = amap(key);
                for i=1:numel(vals)
                    assert(any(strcmp(amap(vals{i}), key)),'Non-symmetric adjacency')
                end
            end
        end
    end
    
    methods(Access = private)
        % Copy constructors
        function obj = copyFromPascalPartsStruct(obj,s)
            nObjects = numel(s);
            obj(nObjects) = PascalObject();  % Preallocate 
            [obj(:).class]      = s(:).class;
            [obj(:).classIndex] = s(:).class_ind;
            [obj(:).mask]       = s(:).mask;
            for i=1:nObjects
                if isempty(s(i).parts)
                    obj(i).parts = [];
                else    % Copy parts
                    obj(i).parts = PascalPart(s(i).parts);
                    [obj(i).parts(:).objectClass] = deal(obj(i).class);
                    partInd = PascalObject.part2index(obj(i).class);
                    for j=1:numel(obj(i).parts)
                        obj(i).parts(j).classIndex = partInd(obj(i).parts(j).class);
                    end
                end
                if ~isempty(obj(i).mask) % Compute bounding box and centroids
                    obj(i).bbox    = mask2bbox(obj(i));
                    obj(i).polygon = mask2polygon(obj(i));
                    %computeCentroids(obj(i));
                end
            end
        end
    end
    methods(Access = protected)
        function obj = copyFromPascalObject(obj, po)
            nObjects = numel(po);
            obj(nObjects) = PascalObject();
            props = properties(PascalObject); props(strcmp(props,'parts')) = [];
            for i=1:numel(props)
                [obj(:).(props{i})] = po(:).(props{i});
            end
            for i=1:nObjects  % Copy parts
                if ~isempty(po(i).parts)
                    obj(i).parts = PascalPart(po(i).parts);
                end
            end
        end        
    end
    
    methods(Hidden)  % Mainly legacy methods
        function obj = PascalObjectLegacy(arg)
            if nargin > 0
                classInd  = PascalObject.class2index();
                nClasses  = numel(unique(cell2mat(classInd.values)));
                if ischar(arg)  % Create an empty object of given class
                    obj.class      = arg;
                    obj.classIndex = PascalObject.class2index(arg);
                elseif isstruct(arg) % Copy from PascalParts struct 
                    readPascalPartsStruct(obj,arg);
                elseif isa(arg, 'PascalObject') % Copy from a PascalObject
                    readPascalObject(obj,arg);
                elseif isstruct(arg) || isa(arg,'PascalObject') 
                    nObj      = length(arg);
                    obj(nObj) = PascalObject(); % Preallocate object array
                    for i=1:nObj
                        obj(i).class      = arg(i).class;
                        obj(i).classIndex = classInd(arg(i).class);
                        obj(i).mask       = arg(i).mask;
                        % Copy parts
                        if isempty(arg(i).parts)
                            obj(i).parts = [];
                        else
                            obj(i).parts = PascalPart(arg(i).parts);
                        end
                        % Copy object class in every part
                        partInd = PascalObject.part2index(obj(i).class);
                        for j=1:numel(obj(i).parts)
                            obj(i).parts(j).objectClass = obj(i).class;
                            obj(i).parts(j).classIndex  = partInd(obj(i).parts(j).class);
                        end
                        % Copy bounding box
                        if (isfield(arg(i),'bbox') || isprop(arg(i),'bbox'))...
                                && ~isempty(arg(i).bbox)
                            obj(i).bbox = arg(i).bbox;
                        elseif ~isempty(obj(i).mask)
                            obj(i).bbox = mask2bbox(obj(i));
                        end
                        % Copy centroid
                        if (isfield(arg(i),'centroid') || isprop(arg(i),'centroid'))...
                                && ~isempty(arg(i).centroid)
                            obj(i).centroid = arg(i).centroid;
                            for j=1:numel(obj(i).parts)
                                obj(i).parts(j).centroid = arg(i).parts(j).centroid;
                            end
                        elseif ~isempty(obj(i).mask)
                            computeCentroids(obj(i));
                        end                        
                        % Copy image name to object and parts
                        if isfield(arg(i),'imname') || isprop(arg(i),'imname') 
                            obj(i).imname = arg(i).imname;
                            if ~isempty(obj(i).parts)
                                [obj(i).parts(:).imname] = deal(obj(i).imname);
                            end
                        end
                        % Check if field values are valid
                        assert(isempty(obj(i).class) || ischar(obj(i).class), ...
                            'Object class must be a string');
                        assert(isempty(obj(i).classIndex) || ...
                            (obj(i).classIndex <= nClasses && obj(i).classIndex > 0),...
                            ['Object class index must be in the range [1,'...
                            num2str(nClasses) ']'])
                        assert(isempty(obj(i).bbox) || numel(obj(i).bbox) == 4, ...
                            'Object bounding box must be a 1x4 vector')
                    end
                else
                    error ('Input must be a struct or a PascalObject object')
                end
            end
        end
        function [partsRef,partsTest] = matchPartsLegacy(objRef,objTest)
            % This is a legacy function and not used any more. Check
            % PascalObject.mathcParts
            
            % we need unnormalized centroids
            if isempty(objRef.centroid) || any(objRef.centroid <= 1)
                computeCentroids(objRef,0);
            end
            if isempty(objTest.centroid) || any(objTest.centroid <= 1)
                computeCentroids(objTest,0);
            end
            
            % keep mutual parts (wrt reference object)
            [isMutualPart,indsTest] = ismember({objRef.parts(:).class},{objTest.parts(:).class});
            partsRef  = objRef.parts(isMutualPart);
            partsTest = objTest.parts(indsTest(indsTest>0));
            partRefNames  = {partsRef(:).class};
            partTestNames = {partsTest(:).class};
            assert(all(strcmp(partRefNames, partTestNames)))
            
            % Parts that appear multiple times in an object (e.g. wheel_1,
            % wheel_2, headlight_1, headlight_2 etc.), are not necessarily
            % matched correctly. We have to explicitly determine if wheel_1
            % in reference object indeed corresponds to wheel_1 in the test
            % object or if we have to re-assign it to another wheel. We do
            % this by minimizing the sum of squared distances of multiple
            % part correspondences for all possible permutations.
            cref  = cat(1,partsRef(:).centroid);    % part centroids
            ctest = cat(1,partsTest(:).centroid);
            % Find multiple parts
            isMultiple = ~cellfun(@isempty,strfind(partRefNames,  '_'));
            multiplePartNames = unique(strtok(partRefNames(isMultiple),'_'));
            for i=1:numel(multiplePartNames)
                inds = find(~cellfun(@isempty,strfind(partRefNames,multiplePartNames{i})));
                if length(inds) > 1 % if there is a single instance skip
                    cr = cref(inds,:);
                    ct = ctest(inds,:);
                    p  = perms(1:size(cr,1))';  % get all possible permutations
                    d  = sum((ct(p,:)-repmat(cr,[size(p,2),1,1])).^2, 2); % point to point distances
                    d  = sum(reshape(d,size(p)));  % sum of point-to-point distances
                    [~, indMinDist] = min(d);
                    partsTest(inds) = partsTest(inds(p(:,indMinDist))); % swap parts
                end
            end
        end
        function cmap = class2colorLegacy()   %TODO: replace this with pascal segmentation colormap
            % CLASS2COLOR  Maps each object class to a color.
            %
            %   cmap = CLASS2COLOR() maps each object class to a
            %       three-element rgb color vector. The colors span the
            %       spectrum in a way so that they are distinguishable from
            %       one-another.
            %   
            %   See also: PascalObject.validObjectClasses, mat2cell
            colors = [
                0.00  0.00  1.00
                0.00  0.50  0.00
                1.00  0.00  0.00
                0.00  0.75  0.75
                0.75  0.00  0.75
                0.75  0.75  0.00
                0.25  0.25  0.25
                0.75  0.25  0.25
                0.95  0.95  0.00
                0.25  0.25  0.75
                0.75  0.75  0.75
                0.00  1.00  0.00
                0.76  0.57  0.17
                0.54  0.63  0.22
                0.34  0.57  0.92
                1.00  0.10  0.60
                0.88  0.75  0.73
                0.10  0.49  0.47
                0.66  0.34  0.65
                0.99  0.41  0.23
                0.00  0.00  1.00];
            classes  = PascalObject.validObjectClasses;
            nClasses = length(classes);
%             colors = varycolor(nClasses);
%             colors = pmkmp(nClasses);
            cmap = containers.Map(classes, ...
                mat2cell(colors(1:nClasses,:), ones(nClasses,1), 3));
            cmap('table') = cmap('diningtable');
        end        
    end
end
