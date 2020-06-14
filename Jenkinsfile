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
				TAG_BUILD = """${sh(returnStdout: true, script: 'bash -c "[[ \\$TAG_NAME ]] && echo ' ' || echo -n dev0+git-${GIT_COMMIT}"')}"""
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
		stage ('Snapify') {
			agent {
				label "scala"
			}
			steps {
				// clean out previously built snaps
				sh "rm *.snap || true"
				sh "snapcraft --use-lxd"
				archiveArtifacts artifacts: '*.snap'
				stash includes: '*.snap', name: 'snapped'
			}
		}
		stage('Deploy') {
			when {
				beforeInput true
				buildingTag()
			}
			input {
				id 'Should-release-nmfu'
				message 'Release this version of NMFU?'
				ok 'Yes, do it!' 
				parameters {
					choice(choices: ['release', 'candidate', 'beta'], description: 'Which release channel should the snap be sent to?', name: 'SNAPCRAFT_RELEASE_CHANNEL')
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
				stage ('Upload Snap') {
					agent {
						label "scala"
					}
					environment {
						SNAP_LOGIN_FILE = credentials('snap-login')
					}
					steps {
						unstash name: 'snapped'
						sh "snapcraft login --with $SNAP_LOGIN_FILE"
						sh "snapcraft upload --release=$SNAPCRAFT_RELEASE_CHANNEL *.snap"
					}
				}
			}
		}
	}
}
