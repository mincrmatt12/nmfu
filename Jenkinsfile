pipeline {
	agent {
		label "docker && linux"
	}
	options {
		preserveStashes(buildCount: 5)
	}
	stages {
		stage ('Build') {
			agent {
				docker { 
					image 'python:3' 
					label "docker && linux"
				}
			}
			environment {
				TAG_BUILD = """${sh(returnStdout: true, script: 'bash -c "[[ \\$TAG_NAME ]] && echo \' \' || echo -n dev0+git-${GIT_COMMIT}"')}"""
			}
			steps {
				sh "rm dist/* || true"
				sh "python setup.py egg_info --tag-build '${TAG_BUILD}' sdist bdist_wheel"
				archiveArtifacts artifacts: 'dist/*'
				stash includes: 'dist/*', name: 'built'
			}
		}
		stage ('Test') {
			agent {
				dockerfile {
					label "docker && linux"
					filename 'Dockerfile.build'
				}
			}
			steps {
				sh "pytest --junit-xml=junit.xml || true"
				junit 'junit.xml'
			}
		}
		stage('Deploy/Package') {
			when {
				beforeInput true
				buildingTag()
			}
			input {
				id 'Should-release-nmfu'
				message 'Release this version of NMFU?'
				ok 'Yes, do it!' 
				parameters {
					choice(choices: ['stable', 'candidate', 'beta'], description: 'Which release channel should the snap be sent to?', name: 'SNAPCRAFT_RELEASE_CHANNEL')
					choice(choices: ['pypi', 'testpypi'], description: 'Which registry should the package be sent to?', name: 'PYPI_REPOSITORY_NAME')
				}
				submitter 'matthew'
			}
			parallel {
				stage ('Upload PyPI') {
					agent {
						dockerfile {
							label "docker && linux"
							filename 'Dockerfile.build'
						}
					}
					environment {
						DPYPI_INFO = credentials('pip-password')
					}
					steps {
						unstash name: 'built'
						sh "twine upload -u $DPYPI_INFO_USR -p $DPYPI_INFO_PSW -r $PYPI_REPOSITORY_NAME --non-interactive dist/*"
					}
				}
				stage('Generate AppImage') {
					agent {
						dockerfile {
							label "docker && linux"
							filename 'Dockerfile.appimage'
						}
					}
					steps {
						sh "appimage-builder"
						archiveArtifacts "*.AppImage"
					}
				}
				stage ('Upload Snap') {
					agent {
						docker {
							label "docker && linux"
							image 'cibuilds/snapcraft:core18'
							args "-u 0:0"
						}
					}
					environment {
						SNAP_LOGIN_FILE = credentials('snapcraft-login')
					}
					steps {
						dir("snapbuild") {
							sh "rm *.snap || true"
							sh "cp ../README.md ../nmfu.py ../setup.py ../snapcraft.yaml ."
							sh "apt-get update && snapcraft"
							archiveArtifacts artifacts: '*.snap'
							sh "snapcraft login --with $SNAP_LOGIN_FILE"
							sh "snapcraft upload --release $SNAPCRAFT_RELEASE_CHANNEL *.snap"
						}
					}
					post {
						always {
							sh "chmod -R a+rw \$PWD/"
						}
					}
				}
			}
		}
	}
}
