classdef OIDAeroplane < PascalObject
    %OIDAEROPLANE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj = OIDAeroplane(arg)
            if nargin > 0
                if isstruct(arg)
                    obj = copyFromOIDStruct(obj,arg);
                elseif isa(arg, 'OIDAeroplane')
                    obj = copyFromPascalObject(obj,arg);
                end
            else
                obj.class = 'aeroplane';
                obj.classIndex = 1;
            end
        end
        
        function obj = crop(obj,box,pad)
            if nargin < 2, box = getbbox(obj); end
            if nargin < 3, pad = 0; end
            for i=1:numel(obj)  % crop polygons
                if isempty(box)
                    bb = getbbox(obj(i));
                else
                    bb = box;
                end
                obj(i).polygon = bsxfun(@minus, obj(i).polygon, bb(1:2)') + pad+1;
                for j=1:numel(obj(i).parts)
                    obj(i).parts(j).polygon = ...
                        bsxfun(@minus, obj(i).parts(j).polygon, bb(1:2)') + pad+1;
                end
            end
            obj = crop@PascalObject(obj, box, pad);
        end
    end
    
    methods(Static)
        function pi = part2index(part)
            switch part
                case 'aeroplane'
                    pi = 1;
                case 'verticalStabilizer'
                    pi = 2;
                case 'nose'
                    pi = 3;
                case 'wing'
                    pi = 4;
                case 'wheel'
                    pi = 5;
                otherwise
                    error('Invalid OID aeroplane part')
            end
        end 
    end
    methods(Static = true, Access = private)
        function p = createPartArray(s, partClass)
            s = s.(partClass);
            nParts = numel(s.id);
            p(nParts) = PascalPart();
            [p(:).class]       = deal(partClass);
            [p(:).classIndex]  = deal(OIDAeroplane.part2index(partClass)); 
            [p(:).objectClass] = deal('aeroplane');
            % Due to memory fragmentation it is impractical to store the
            % masks for all the objects and parts of the dataset. For that
            % reason, we store the polygons and compute the masks when needed.
            for i=1:nParts  
                x = s.polygon{i}(1,:); y = s.polygon{i}(2,:);
                p(i).bbox = [floor(min(x)), floor(min(y)), ceil(max(x)), ceil(max(y))];
                p(i).polygon = [x; y];
                % TODO: Compute centroids? maybe it will be too slow
            end
        end
    end
    methods(Access = private)
        function obj = copyFromOIDStruct(obj, oid)
            % TODO: add support for 'ignore' flags
            nAeroplanes = numel(oid.aeroplane.id);
            obj(nAeroplanes) = OIDAeroplane();
            [obj(:).class] = deal('aeroplane');
            [obj(:).classIndex] = deal(1);
            vStab = OIDAeroplane.createPartArray(oid, 'verticalStabilizer');
            nose  = OIDAeroplane.createPartArray(oid, 'nose');
            wing  = OIDAeroplane.createPartArray(oid, 'wing');
            wheel = OIDAeroplane.createPartArray(oid, 'wheel');
            aeroPolygon   = oid.aeroplane.polygon;
            aeroplaneId   = oid.aeroplane.id;
            vStabParentId = oid.verticalStabilizer.parentId;
            noseParentId  = oid.nose.parentId;
            wingParentId  = oid.wing.parentId;
            wheelParentId = oid.wheel.parentId;
            for i=1:nAeroplanes;
                x = aeroPolygon{i}(1,:); y = aeroPolygon{i}(2,:);
                obj(i).polygon = [x; y];
                obj(i).bbox = [floor(min(x)), floor(min(y)), ceil(max(x)), ceil(max(y))];
                obj(i).parts = [vStab(vStabParentId == aeroplaneId(i)),...
                    nose(noseParentId == aeroplaneId(i)),...
                    wing(wingParentId == aeroplaneId(i)),...
                    wheel(wheelParentId == aeroplaneId(i))];
            end
        end
    end
end
