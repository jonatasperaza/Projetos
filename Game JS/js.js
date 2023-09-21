        // Configuração do Three.js
        const scene = new THREE.Scene();

        // Jogador 1
        const player1Size = 1;
        const player1Geometry = new THREE.BoxGeometry(player1Size, player1Size, player1Size);
        const player1Material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
        const player1 = new THREE.Mesh(player1Geometry, player1Material);
        scene.add(player1);
        player1.position.set(-2, 0, 0);

        // Câmera do jogador 1
        const camera1 = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        player1.add(camera1);
        camera1.position.set(0, 1, -2);
        player1.rotation.order = 'YXZ'; // Importante para a rotação correta

        // Jogador 2
        const player2Size = 1;
        const player2Geometry = new THREE.BoxGeometry(player2Size, player2Size, player2Size);
        const player2Material = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        const player2 = new THREE.Mesh(player2Geometry, player2Material);
        scene.add(player2);
        player2.position.set(2, 0, 0);

        // Câmera do jogador 2
        const camera2 = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        player2.add(camera2);
        camera2.position.set(0, 1, -2);
        player2.rotation.order = 'YXZ'; // Importante para a rotação correta

        // Controles do jogador 1 (teclas WASD)
        const player1Speed = 0.1;

        document.addEventListener("keydown", (event) => {
            switch (event.key) {
                case "w":
                    player1.position.z -= player1Speed;
                    break;
                case "s":
                    player1.position.z += player1Speed;
                    break;
                case "a":
                    player1.position.x -= player1Speed;
                    break;
                case "d":
                    player1.position.x += player1Speed;
                    break;
            }
        });

        // Controles do jogador 2 (setas)
        const player2Speed = 0.1;

        document.addEventListener("keydown", (event) => {
            switch (event.key) {
                case "ArrowUp":
                    player2.position.z -= player2Speed;
                    break;
                case "ArrowDown":
                    player2.position.z += player2Speed;
                    break;
                case "ArrowLeft":
                    player2.position.x -= player2Speed;
                    break;
                case "ArrowRight":
                    player2.position.x += player2Speed;
                    break;
            }
        });

        // Captura de movimento do mouse para a câmera do jogador 1
        const mouseSensitivity = 0.005;
        let player1Yaw = 0;
        let player1Pitch = 0;

        document.addEventListener("mousemove", (event) => {
            player1Yaw += event.movementX * mouseSensitivity;
            player1Pitch += event.movementY * mouseSensitivity;

            // Limitar a inclinação da câmera para evitar rotação excessiva
            player1Pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, player1Pitch));

            player1.rotation.y = player1Yaw;
            player1.rotation.x = player1Pitch;
        });

        // Configuração da cena e renderização
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 5, 10);
        camera.lookAt(0, 0, 0);

        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        // Ajuste a janela para tela cheia
        window.addEventListener("resize", () => {
            const newWidth = window.innerWidth;
            const newHeight = window.innerHeight;
            renderer.setSize(newWidth, newHeight);
            camera.aspect = newWidth / newHeight;
            camera.updateProjectionMatrix();
        });

        // Função de renderização
        const animate = function () {
            requestAnimationFrame(animate);

            // Atualize as câmeras dos jogadores para que sigam seus cubos correspondentes
            camera1.position.copy(player1.position);
            camera2.position.copy(player2.position);

            renderer.render(scene, camera);
        };

        // Iniciar o jogo
        animate();